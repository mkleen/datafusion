// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Checks that float keys holding both `-0.0` and `0.0` give the same results
//! under every execution strategy, see #26138.
//!
//! Each query runs on a float key column under a matrix of settings (hash vs
//! sort-merge join, partitions, batch sizes, sorted input) and is compared with
//! the same query on `ki`, an integer column that maps each float to its place
//! in SQL order, with `-0.0` and `0.0` mapping to the same integer.
//! `float_key_semantics.slt` holds the individual cases.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Float32Array, Float64Array, Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::MemTable;
use datafusion::prelude::*;
use datafusion_common::test_util::batches_to_sort_string;

/// Key values and their SQL-order class, `None` for NULL.
const DOMAIN: [(Option<f64>, Option<i32>); 6] = [
    (Some(-0.0), Some(1)),
    (Some(0.0), Some(1)),
    (Some(-1.0), Some(0)),
    (Some(1.0), Some(2)),
    (Some(2.5), Some(3)),
    (None, None),
];

const ROWS: i32 = 60;

/// Query templates. `{k}` is replaced with the key column. Queries return
/// only columns that are the same for the float and the reference run.
const QUERIES: &[(&str, bool, &str)] = &[
    // (name, ordered, sql)
    (
        "join_k",
        false,
        "SELECT t.id, u.id AS uid FROM t JOIN u ON t.{k} = u.{k}",
    ),
    (
        "join_k_b",
        false,
        "SELECT t.id, u.id AS uid FROM t JOIN u ON t.{k} = u.{k} AND t.b = u.b",
    ),
    (
        "join_list",
        false,
        "SELECT t.id, u.id AS uid FROM t JOIN u \
         ON make_array(t.{k}, t.b) = make_array(u.{k}, u.b)",
    ),
    (
        "group_by_k_b",
        false,
        "SELECT min(ki) AS c, b, count(*) AS n, sum(id) AS s FROM t GROUP BY {k}, b",
    ),
    (
        "group_by_list",
        false,
        "SELECT min(ki) AS c, min(b) AS b, count(*) AS n, sum(id) AS s FROM t \
         GROUP BY make_array({k}, b)",
    ),
    (
        "group_by_struct",
        false,
        "SELECT min(ki) AS c, min(b) AS b, count(*) AS n, sum(id) AS s FROM t \
         GROUP BY named_struct('k', {k}, 'b', b)",
    ),
    (
        "distinct",
        false,
        "SELECT count(*) AS n FROM (SELECT DISTINCT {k}, b FROM t)",
    ),
    ("order_by", true, "SELECT id FROM t ORDER BY {k}, b, id"),
    (
        "order_by_limit",
        true,
        "SELECT id FROM t ORDER BY {k}, b, id LIMIT 25",
    ),
    (
        "rank",
        false,
        "SELECT id, rank() OVER (ORDER BY {k}) AS r FROM t",
    ),
    (
        "row_number_partition",
        false,
        "SELECT id, row_number() OVER (PARTITION BY {k} ORDER BY id) AS r FROM t",
    ),
];

/// Queries that are wrong on main today, with the issue tracking each. A fix
/// removes its entry; the test fails if the list and the results disagree.
const KNOWN_WRONG: &[(&str, &str)] = &[
    ("join_k_b", "#26115"),
    ("join_list", "#26115"),
    ("order_by", "#26138"),
    ("order_by_limit", "#26138"),
    ("rank", "#26138"),
    ("row_number_partition", "#25766"),
];

#[derive(Debug, Clone, Copy)]
struct Variant {
    prefer_hash_join: bool,
    target_partitions: usize,
    batch_size: usize,
    sorted: bool,
}

fn variants() -> Vec<Variant> {
    let mut variants = vec![];
    for prefer_hash_join in [true, false] {
        for target_partitions in [1, 4] {
            for batch_size in [1, 7, 8192] {
                for sorted in [false, true] {
                    variants.push(Variant {
                        prefer_hash_join,
                        target_partitions,
                        batch_size,
                        sorted,
                    });
                }
            }
        }
    }
    variants
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("k", DataType::Float64, true),
        Field::new("k32", DataType::Float32, true),
        Field::new("ki", DataType::Int32, true),
        Field::new("b", DataType::Int32, false),
    ]))
}

/// `(id, k, ki, b)` rows. `stride` spreads the keys differently per table.
/// With `with_neg_zero = false`, `-0.0` is replaced by `0.0`.
fn rows(stride: i32, with_neg_zero: bool) -> Vec<(i32, Option<f64>, Option<i32>, i32)> {
    (0..ROWS)
        .map(|id| {
            let (k, ki) = DOMAIN[((id * stride) % DOMAIN.len() as i32) as usize];
            let k = k.map(|v| if with_neg_zero { v } else { v + 0.0 });
            (id, k, ki, (id / 6) % 3)
        })
        .collect()
}

/// Splits `rows` into two partitions of uneven batches. With `sorted`, each
/// partition is sorted by `k` (as Arrow sorts it), then `id`.
fn partitions(
    mut rows: Vec<(i32, Option<f64>, Option<i32>, i32)>,
    sorted: bool,
) -> Vec<Vec<RecordBatch>> {
    if sorted {
        rows.sort_by(|l, r| match (l.1, r.1) {
            (Some(a), Some(b)) => a.total_cmp(&b).then(l.0.cmp(&r.0)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => l.0.cmp(&r.0),
        });
    }
    (0..2)
        .map(|p| {
            let part: Vec<_> = rows.iter().filter(|r| r.0 % 2 == p).collect();
            part.chunks(if p == 0 { 7 } else { 11 })
                .map(|chunk| {
                    RecordBatch::try_new(
                        schema(),
                        vec![
                            Arc::new(Int32Array::from_iter_values(
                                chunk.iter().map(|r| r.0),
                            )),
                            Arc::new(Float64Array::from_iter(chunk.iter().map(|r| r.1))),
                            Arc::new(Float32Array::from_iter(
                                chunk.iter().map(|r| r.1.map(|v| v as f32)),
                            )),
                            Arc::new(Int32Array::from_iter(chunk.iter().map(|r| r.2))),
                            Arc::new(Int32Array::from_iter_values(
                                chunk.iter().map(|r| r.3),
                            )),
                        ],
                    )
                    .unwrap()
                })
                .collect()
        })
        .collect()
}

fn context(variant: Variant, with_neg_zero: bool) -> SessionContext {
    let config = SessionConfig::new()
        .with_target_partitions(variant.target_partitions)
        .with_batch_size(variant.batch_size)
        .set_bool(
            "datafusion.optimizer.prefer_hash_join",
            variant.prefer_hash_join,
        );
    let ctx = SessionContext::new_with_config(config);
    for (name, stride) in [("t", 1), ("u", 5)] {
        let mut table = MemTable::try_new(
            schema(),
            partitions(rows(stride, with_neg_zero), variant.sorted),
        )
        .unwrap();
        if variant.sorted {
            table = table.with_sort_order(vec![
                vec![col("k").sort(true, false)],
                vec![col("k32").sort(true, false)],
            ]);
        }
        ctx.register_table(name, Arc::new(table)).unwrap();
    }
    ctx
}

async fn run(ctx: &SessionContext, sql: &str, ordered: bool) -> String {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    if ordered {
        pretty_format_batches(&batches).unwrap().to_string()
    } else {
        batches_to_sort_string(&batches)
    }
}

/// Runs every query under every variant and returns the failing queries,
/// each with the first variant and key column it failed for.
async fn failing_queries(with_neg_zero: bool) -> BTreeMap<&'static str, String> {
    let reference = context(
        Variant {
            prefer_hash_join: true,
            target_partitions: 1,
            batch_size: 8192,
            sorted: false,
        },
        with_neg_zero,
    );
    let contexts: Vec<_> = variants()
        .into_iter()
        .map(|v| (v, context(v, with_neg_zero)))
        .collect();

    let mut failing = BTreeMap::new();
    for &(name, ordered, sql) in QUERIES {
        let expected = run(&reference, &sql.replace("{k}", "ki"), ordered).await;
        'query: for key in ["k", "k32"] {
            let sql = sql.replace("{k}", key);
            for (variant, ctx) in &contexts {
                let actual = run(ctx, &sql, ordered).await;
                if actual != expected {
                    failing.insert(
                        name,
                        format!(
                            "{name} on `{key}` with {variant:?}\n{sql}\n\
                             expected:\n{expected}\nactual:\n{actual}"
                        ),
                    );
                    break 'query;
                }
            }
        }
    }
    failing
}

#[tokio::test]
async fn float_key_semantics_match_reference() {
    let failing = failing_queries(true).await;
    let known: BTreeMap<_, _> = KNOWN_WRONG.iter().copied().collect();

    let regressed: Vec<_> = failing
        .iter()
        .filter(|(name, _)| !known.contains_key(*name))
        .map(|(_, details)| details.as_str())
        .collect();
    let fixed: Vec<_> = known
        .iter()
        .filter(|(name, _)| !failing.contains_key(*name))
        .map(|(name, issue)| format!("{name} ({issue})"))
        .collect();

    assert!(
        regressed.is_empty() && fixed.is_empty(),
        "New failures:\n{}\n\nNow passing, remove from KNOWN_WRONG: {fixed:?}",
        regressed.join("\n\n"),
    );
}

/// Without `-0.0` in the data every query must match the reference, which
/// shows the queries and the reference mapping are right.
#[tokio::test]
async fn float_key_semantics_self_check() {
    let failing = failing_queries(false).await;
    assert!(
        failing.is_empty(),
        "{}",
        failing.into_values().collect::<Vec<_>>().join("\n\n")
    );
}
