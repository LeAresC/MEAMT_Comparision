import contextlib
import io
import sys
import types
import unittest

import numpy as np
from deap import base, creator, tools


try:
    import moeabench  # noqa: F401
except ImportError:
    moeabench = types.ModuleType("moeabench")
    moeabench.moeas = types.SimpleNamespace(BaseMoea=object)
    progress = types.ModuleType("moeabench.progress")
    progress.get_active_pbar = lambda: None
    sys.modules["moeabench"] = moeabench
    sys.modules["moeabench.progress"] = progress


from src.meamt_core_ndom import (  # noqa: E402
    build_toolbox,
    fast_clone,
    gen_inicial_tables,
    run,
    sel_nsga2,
    setup_deap_classes,
)
from src.meamt_ndom import (  # noqa: E402
    _normalize_constraints,
    _normalize_objectives,
)


class DeapClassesMixin:
    def clean_creator(self):
        for name in ("FitnessMin", "Individual", "SubPopulation"):
            if hasattr(creator, name):
                delattr(creator, name)

    def setUp(self):
        self.clean_creator()
        setup_deap_classes(2)

    def tearDown(self):
        self.clean_creator()

    @staticmethod
    def individual(values, cv=0.0, genes=(0.0, 0.0)):
        ind = creator.Individual(genes)
        ind.fitness.values = values
        ind.fitness.constraint_violation = cv
        return ind


class ConstraintFitnessTests(DeapClassesMixin, unittest.TestCase):
    def test_feasible_dominates_infeasible_regardless_of_objectives(self):
        feasible = self.individual((10.0, 10.0), 0.0)
        infeasible = self.individual((1.0, 1.0), 0.1)
        self.assertTrue(feasible.fitness.dominates(infeasible.fitness))
        self.assertFalse(infeasible.fitness.dominates(feasible.fitness))

    def test_lower_violation_dominates_and_equal_violation_does_not(self):
        lower = self.individual((10.0, 10.0), 0.1)
        higher = self.individual((1.0, 1.0), 0.5)
        equal = self.individual((0.0, 0.0), 0.1)
        self.assertTrue(lower.fitness.dominates(higher.fitness))
        self.assertFalse(higher.fitness.dominates(lower.fitness))
        self.assertFalse(lower.fitness.dominates(equal.fitness))
        self.assertFalse(equal.fitness.dominates(lower.fitness))

    def test_feasible_dominance_matches_deap_and_respects_obj(self):
        first = self.individual((1.0, 5.0))
        second = self.individual((2.0, 1.0))

        class ReferenceFitness(base.Fitness):
            weights = (-1.0, -1.0)

        ref_first = ReferenceFitness(first.fitness.values)
        ref_second = ReferenceFitness(second.fitness.values)
        self.assertEqual(
            first.fitness.dominates(second.fitness),
            ref_first.dominates(ref_second),
        )
        self.assertTrue(first.fitness.dominates(second.fitness, obj=slice(0, 1)))

    def test_equality_and_sorting_distinguish_constraint_violation(self):
        feasible = self.individual((1.0, 1.0), 0.0)
        infeasible = self.individual((1.0, 1.0), 0.2)
        self.assertNotEqual(feasible.fitness, infeasible.fitness)
        mapping = {feasible.fitness: "feasible", infeasible.fitness: "infeasible"}
        self.assertEqual(len(mapping), 2)
        front = tools.sortNondominated(
            [feasible, infeasible], 2, first_front_only=True
        )[0]
        self.assertEqual(front, [feasible])

    def test_clone_preserves_values_violation_and_parent(self):
        original = self.individual((1.0, 2.0), 0.37)
        original.Parent_Table = 2
        clone = fast_clone(original)
        self.assertEqual(clone.fitness.values, original.fitness.values)
        self.assertEqual(clone.fitness.constraint_violation, 0.37)
        self.assertEqual(clone.Parent_Table, 2)

    def test_partial_table_prefers_feasible_and_restores_fitness(self):
        feasible = self.individual((10.0, 20.0), 0.0)
        infeasible = self.individual((1.0, 2.0), 0.1)
        original_values = [ind.fitness.values for ind in (feasible, infeasible)]
        original_cv = [ind.fitness.constraint_violation for ind in (feasible, infeasible)]
        selected = sel_nsga2([feasible, infeasible], 1, 1, 2)
        self.assertEqual(selected, [feasible])
        self.assertEqual(
            [ind.fitness.values for ind in (feasible, infeasible)], original_values
        )
        self.assertEqual(
            [ind.fitness.constraint_violation for ind in (feasible, infeasible)],
            original_cv,
        )


class NormalizationTests(unittest.TestCase):
    def test_constraint_shapes_and_cv(self):
        cases = [
            (0.3, 1, (1, 1), [0.3]),
            ([-0.2, 0.0, 0.4], 3, (3, 1), [0.0, 0.0, 0.4]),
            ([[-0.2], [0.0], [0.4]], 3, (3, 1), [0.0, 0.0, 0.4]),
            ([[-0.2, 0.3, 0.4]], 1, (1, 3), [0.7]),
            ([[-0.2, 0.0, 0.4]], 3, (3, 1), [0.0, 0.0, 0.4]),
        ]
        for values, count, shape, expected_cv in cases:
            with self.subTest(values=values, count=count):
                normalized = _normalize_constraints(values, count)
                self.assertEqual(normalized.shape, shape)
                np.testing.assert_allclose(
                    np.maximum(normalized, 0.0).sum(axis=1), expected_cv
                )

    def test_multiple_constraints_sum_positive_parts(self):
        normalized = _normalize_constraints(
            [[-0.3, 0.2, 0.0], [0.1, 0.4, -0.2]], 2
        )
        np.testing.assert_allclose(
            np.maximum(normalized, 0.0).sum(axis=1), [0.2, 0.5]
        )

    def test_invalid_constraints_raise(self):
        for values, count in (
            (0.2, 2),
            ([0.1, 0.2], 3),
            ([], 1),
            ([[[0.1]]], 1),
            ([np.nan], 1),
            ([np.inf], 1),
            ([-np.inf], 1),
        ):
            with self.subTest(values=values, count=count):
                with self.assertRaises(ValueError):
                    _normalize_constraints(values, count)

    def test_objective_normalization(self):
        result = _normalize_objectives([1.0, 2.0], 1, 2)
        self.assertEqual(result.shape, (1, 2))
        for values, count, n_obj in (
            ([1.0, 2.0], 2, 2),
            ([[1.0], [2.0]], 2, 2),
            ([[1.0, np.nan]], 1, 2),
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    _normalize_objectives(values, count, n_obj)


class EvolutionTests(DeapClassesMixin, unittest.TestCase):
    def test_archive_initialization_deduplicates_and_snapshots_are_temporal(self):
        toolbox = build_toolbox(lambda ind: tuple(ind), 2, 8, 2)
        toolbox.register("map", map)
        population = toolbox.population()
        for index, ind in enumerate(population):
            ind[:] = [index / 10.0, (7 - index) / 10.0]
            ind.fitness.values = tuple(ind)
            ind.fitness.constraint_violation = 0.0

        table_sizes = [0, 3, 2, 3]
        tables = gen_inicial_tables(population, 4, table_sizes, 2)
        expected_unique = len(
            {id(ind) for table_id in range(1, 4) for ind in tables[table_id]}
        )
        snapshots = []

        def snapshot(current_tables):
            snapshots.append(
                {
                    id(ind)
                    for table_id in range(1, 4)
                    for ind in current_tables[table_id]
                }
            )

        with contextlib.redirect_stdout(io.StringIO()):
            result = run(
                tables=tables,
                num_tables=4,
                pop_size=8,
                ngen=2,
                max_table_size=table_sizes,
                toolbox=toolbox,
                cxpb=0.0,
                mutpb=1.0,
                n_obj=2,
                snapshot_callback=snapshot,
            )

        self.assertEqual(len(snapshots), 3)
        self.assertEqual(len(snapshots[0]), expected_unique)
        self.assertEqual(len({id(ind) for ind in result[0]}), len(result[0]))


if __name__ == "__main__":
    unittest.main()
