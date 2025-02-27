//! Useful tests for debugging since they are hand-written and easy to see the debugging output.

use hnsw::{Hnsw, Searcher};
use itertools::Itertools;
use rand_pcg::Pcg64;
use space::{Metric, Neighbor};

struct Euclidean;

impl Metric<Vec<f64>> for Euclidean {
    type Unit = u64;
    fn distance(&self, a: &Vec<f64>, b: &Vec<f64>) -> u64 {
        a.iter()
            .zip(b.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
            .to_bits()
    }
}

struct TestBruteForceHelper {
    vectors: Vec<(usize, Vec<f64>)>,
}

impl TestBruteForceHelper {
    fn new() -> Self {
        Self {
            vectors: Vec::new(),
        }
    }

    fn push(&mut self, v: (usize, Vec<f64>)) {
        self.vectors.push(v);
    }

    fn search(&self, query: &Vec<f64>, top_k: usize) -> Vec<usize> {
        let metric = Euclidean;
        let mut candidates: Vec<(usize, u64)> = self
            .vectors
            .iter()
            .map(|v| (v.0.clone(), metric.distance(&query, &v.1)))
            .collect_vec();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        candidates.into_iter().take(top_k).map(|v| v.0).collect()
    }
}

fn test_hnsw() -> (
    Hnsw<Euclidean, Vec<f64>, Pcg64, 12, 24>,
    Searcher<u64>,
    TestBruteForceHelper,
) {
    let mut searcher = Searcher::default();
    let mut hnsw = Hnsw::new(Euclidean);
    let mut helper = TestBruteForceHelper::new();
    let features = vec![
        &[0.0, 0.0, 0.0, 1.0],
        &[0.0, 0.0, 1.0, 0.0],
        &[0.0, 1.0, 0.0, 0.0],
        &[1.0, 0.0, 0.0, 0.0],
        &[0.0, 0.0, 1.0, 1.0],
        &[0.0, 1.0, 1.0, 0.0],
        &[1.0, 1.0, 0.0, 0.0],
        &[1.0, 0.0, 0.0, 1.0],
    ];

    for (index, feature) in features.iter().enumerate() {
        helper.push((index, feature.to_vec()));
        hnsw.insert(feature.to_vec(), &mut searcher);
    }

    (hnsw, searcher, helper)
}

#[test]
fn insertion() {
    test_hnsw();
}

#[test]
fn nearest_neighbor() {
    let (hnsw, mut searcher, helper) = test_hnsw();
    let searcher = &mut searcher;
    let mut neighbors = vec![
        Neighbor {
            index: !0,
            distance: !0,
        };
        8
    ];

    hnsw.nearest(&vec![0.0, 0.0, 0.0, 1.0], 24, searcher, &mut neighbors);
    // Distance 1
    neighbors[1..3].sort_unstable();
    // Distance 2
    neighbors[3..6].sort_unstable();
    // Distance 3
    neighbors[6..8].sort_unstable();
    assert_eq!(
        neighbors,
        [
            Neighbor {
                index: 0,
                distance: 0
            },
            Neighbor {
                index: 4,
                distance: 4607182418800017408
            },
            Neighbor {
                index: 7,
                distance: 4607182418800017408
            },
            Neighbor {
                index: 1,
                distance: 4609047870845172685
            },
            Neighbor {
                index: 2,
                distance: 4609047870845172685
            },
            Neighbor {
                index: 3,
                distance: 4609047870845172685
            },
            Neighbor {
                index: 5,
                distance: 4610479282544200874
            },
            Neighbor {
                index: 6,
                distance: 4610479282544200874
            }
        ]
    );
    // test for not panicking
    for topk in 0..8 {
        let mut neighbors = vec![
            Neighbor {
                index: !0,
                distance: !0,
            };
            topk
        ];
        hnsw.nearest(&vec![0.0, 0.0, 0.0, 1.0], 24, searcher, &mut neighbors);
        let result = neighbors.iter().map(|item| item.index).collect_vec();
        assert_eq!(result, helper.search(&vec![0.0, 0.0, 0.0, 1.0], topk));
    }

    let mut searcher = Searcher::default();
    let mut hnsw: Hnsw<Euclidean, Vec<f64>, Pcg64, 12, 24> = Hnsw::new(Euclidean);
    for all_vec in 2..300 {
        hnsw.insert(
            vec![0.0, 1_f64 / all_vec as f64, 0.0, 1_f64 / all_vec as f64],
            &mut searcher,
        );
        for topk in 1..all_vec {
            let mut neighbors = vec![
                Neighbor {
                    index: !0,
                    distance: !0,
                };
                topk
            ];
            hnsw.nearest(
                &vec![0.0, 0.0, 0.0, 1.0],
                topk,
                &mut searcher,
                &mut neighbors,
            );
        }
    }
}
