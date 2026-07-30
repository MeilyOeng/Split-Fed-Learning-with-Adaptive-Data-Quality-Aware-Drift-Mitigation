from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest


class FrozenServerTests(unittest.TestCase):
    def test_frozen_routes_and_parameter_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env.update({
                "FROZEN_VALIDATION_MODE": "true", "GPU_ENABLED": "false", "OUTPUT_DIR": directory,
                "MODEL_HIDDEN_SIZE": "4", "MODEL_NUM_LAYERS": "2", "MODEL_SPLIT_LAYER": "1",
                "MODEL_INPUT_SIZE": "1", "MODEL_OUTPUT_SIZE": "1", "MODEL_DROPOUT": "0",
            })
            script = r'''
import json
import torch
import server.server as module
before = module.state.parameter_hash()
client = module.app.test_client()
round_status = client.post("/round/start", json={"expected_clients": ["1"]}).status_code
train_status = client.post("/train_step", data=b"").status_code
lstm = next(item for item in module.state.model.modules() if isinstance(item, torch.nn.LSTM))
result = module.state.inference_step("1", torch.zeros(2, lstm.input_size))
after = module.state.parameter_hash()
print(json.dumps({"before": before, "after": after, "round": round_status, "train": train_status, "grad": result["gradients_enabled"]}))
'''
            completed = subprocess.run(
                ["./venv/bin/python", "-c", script], cwd=os.getcwd(), env=env,
                check=True, capture_output=True, text=True,
            )
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertEqual(payload["round"], 409)
            self.assertEqual(payload["train"], 409)
            self.assertFalse(payload["grad"])
            self.assertEqual(payload["before"], payload["after"])
