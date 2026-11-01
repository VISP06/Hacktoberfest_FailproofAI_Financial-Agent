import os

os.environ["FAILPROOF_ENV"] = "test"

import failproofai_sdk as fp

fp.configure(environment="test")
