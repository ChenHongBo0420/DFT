"""Fast repository and CLI tests using only the Python standard library."""

from contextlib import redirect_stderr
import csv
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

from dftpy.cli import _output_directory, check_inputs, parse_args
from dftpy.paths import find_poscar, read_file_list, structure_folder

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_splits", ROOT / "scripts/build_splits.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PathTests(unittest.TestCase):
    def test_csv_paths_are_relative_to_csv_not_cwd(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lists = root / "lists"
            lists.mkdir()
            manifest = lists / "train.csv"
            with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.writer(stream)
                writer.writerow(["files"])
                writer.writerow(["../sample, one"])
                writer.writerow([str(root / "absolute")])
                writer.writerow([""])
            self.assertEqual(read_file_list(manifest), [str(root / "sample, one"), str(root / "absolute")])

    def test_poscar_layouts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plain = root / "plain"
            plain.mkdir()
            poscar = plain / "POSCAR"
            poscar.touch()
            self.assertEqual(find_poscar(plain), poscar)
            self.assertEqual(find_poscar(poscar), poscar)
            nested = root / "nested" / "POSCAR"
            nested.mkdir(parents=True)
            (nested / "POSCAR").touch()
            self.assertEqual(structure_folder(nested), root / "nested")
            self.assertEqual(find_poscar(root / "nested"), nested / "POSCAR")
            with self.assertRaises(FileNotFoundError):
                find_poscar(root / "missing")

    def test_bad_csv_header_and_empty_list(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "list.csv"
            path.write_text("wrong\na\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_file_list(path)
            path.write_text("files\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_file_list(path)

    def test_all_example_lists_resolve(self):
        for directory in ("DFT_CSVs", "DFT_CSVs_v2"):
            for manifest in (ROOT / directory).glob("*.csv"):
                column = "file_loc_test" if manifest.name == "predict.csv" else "files"
                for sample in read_file_list(manifest, column):
                    self.assertTrue(find_poscar(sample).is_file())

    def test_task_file_requirements(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                check_inputs([temporary], "chg")
            for name in ("energy", "forces", "stress"):
                (Path(temporary) / name).touch()
            check_inputs([temporary], "energy")


class CommandTests(unittest.TestCase):
    def test_defaults_follow_original_input_configuration(self):
        args = parse_args(["train", "--task", "dos", "--train-list", "train.csv", "--val-list", "val.csv"])
        self.assertEqual(args.num_gamma, 18)
        self.assertEqual(args.cut_off_rad, 5.0)

    def test_invalid_arguments(self):
        cases = [
            ["infer", "--infer-list", "x.csv"],
            ["infer", "--infer-list", "x.csv", "--predict-dos", "--padding-multiplier", "0.5"],
            ["infer", "--infer-list", "x.csv", "--predict-dos", "--cut-off-rad", "nan"],
        ]
        for argv in cases:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)

    def test_output_directory_restores_cwd_and_refuses_overwrite(self):
        args = parse_args(["check-data", "--list", "x.csv"])
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with self.assertRaisesRegex(RuntimeError, "sample error"):
                with _output_directory(output, args):
                    self.assertEqual(Path.cwd(), output)
                    raise RuntimeError("sample error")
            self.assertEqual(Path.cwd(), original)
            self.assertTrue((output / "run.json").is_file())
            with self.assertRaises(FileExistsError):
                with _output_directory(output, args):
                    pass


class SplitTests(unittest.TestCase):
    def test_reproducible_disjoint_splits(self):
        samples = [Path(str(i)) for i in range(19)]
        first = builder.split_samples(samples, 42)
        self.assertEqual(first, builder.split_samples(list(reversed(samples)), 42))
        self.assertEqual(set().union(*map(set, first)), set(samples))
        self.assertEqual(sum(map(len, first)), len(samples))
        self.assertTrue(all(first))

    def test_too_few_samples(self):
        with self.assertRaises(ValueError):
            builder.split_samples([Path("one")], 42)

    def test_split_writer_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "test.csv"
            samples = [root / "sample"]
            builder.write_list(path, samples)
            self.assertEqual(read_file_list(path), [str(samples[0])])


if __name__ == "__main__":
    unittest.main()
