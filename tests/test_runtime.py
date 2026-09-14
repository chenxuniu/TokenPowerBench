import json
import unittest
from unittest.mock import patch

from tokenpowerbench import runtime


class RuntimeIdentityTests(unittest.TestCase):
    def identity(self, uid, euid, gid, egid):
        with patch.object(runtime.os, "getuid", return_value=uid, create=True), \
             patch.object(runtime.os, "geteuid", return_value=euid, create=True), \
             patch.object(runtime.os, "getgid", return_value=gid, create=True), \
             patch.object(runtime.os, "getegid", return_value=egid, create=True):
            return runtime.runtime_identity()

    def test_effective_root_with_nonroot_real_uid(self):
        identity = self.identity(uid=1000, euid=0, gid=1001, egid=0)
        self.assertTrue(identity["is_root"])
        self.assertEqual(identity["uid"], 1000)
        self.assertEqual(identity["euid"], 0)
        self.assertEqual(identity["gid"], 1001)
        self.assertEqual(identity["egid"], 0)
        self.assertEqual(json.loads(json.dumps(identity)), identity)

    def test_real_root_with_unprivileged_effective_uid(self):
        identity = self.identity(uid=0, euid=1000, gid=0, egid=1001)
        self.assertFalse(identity["is_root"])

    def test_nonroot(self):
        self.assertFalse(self.identity(1000, 1000, 1000, 1000)["is_root"])

    def test_unix_identifiers_unavailable(self):
        with patch.multiple(runtime.os, getuid=None, geteuid=None, getgid=None, getegid=None, create=True):
            identity = runtime.runtime_identity()
        for field in ("uid", "euid", "gid", "egid", "is_root"):
            self.assertIsNone(identity[field])

    def test_effective_uid_unavailable_does_not_assume_real_uid(self):
        with patch.object(runtime.os, "getuid", return_value=0, create=True), \
             patch.object(runtime.os, "geteuid", side_effect=NotImplementedError, create=True):
            identity = runtime.runtime_identity()
        self.assertEqual(identity["uid"], 0)
        self.assertIsNone(identity["euid"])
        self.assertIsNone(identity["is_root"])

    def test_identifier_os_error_becomes_unknown(self):
        with patch.object(runtime.os, "geteuid", side_effect=OSError("unavailable"), create=True):
            self.assertIsNone(runtime.runtime_identity()["is_root"])

    def test_platform_is_recorded_without_username_or_hostname(self):
        with patch.object(runtime.platform, "system", return_value="Linux"), \
             patch.object(runtime.platform, "machine", return_value="aarch64"):
            identity = runtime.runtime_identity()
        self.assertEqual(identity["platform"], "Linux")
        self.assertEqual(identity["machine_architecture"], "aarch64")
        self.assertEqual(set(identity), {"uid", "euid", "gid", "egid", "is_root", "platform", "machine_architecture"})


if __name__ == "__main__":
    unittest.main()
