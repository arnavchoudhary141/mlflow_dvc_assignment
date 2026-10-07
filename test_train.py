import unittest
import os
import subprocess


class TestMLflowDVCProject(unittest.TestCase):

    def test_train_file_exists(self):
        """Check that train.py exists."""
        self.assertTrue(os.path.exists("train.py"))

    def test_dvc_file_exists(self):
        """Check that the DVC metadata file exists."""
        self.assertTrue(os.path.exists("data/train.csv.dvc"))

    def test_training_pipeline(self):
        """Run the training pipeline and verify successful execution."""
        result = subprocess.run(
            ["python", "train.py"],
            capture_output=True,
            text=True
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Experiment tracked in MLflow!", result.stdout)

    def test_mlflow_artifact(self):
        """Check that the MLflow artifact is generated."""
        self.assertTrue(os.path.exists("mlflow_artifact.txt"))


if __name__ == "__main__":
    unittest.main()