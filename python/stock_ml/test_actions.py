import os
from datetime import datetime,timedelta,timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,MagicMock
import numpy as np
import actions_runner as actions
from diagnostics import safe_error


class ActionsTests(unittest.TestCase):
    def test_due_times_are_based_on_completed_work_not_wall_clock_modulo(self):
        now=datetime(2026,11,24,tzinfo=timezone.utc)
        self.assertTrue(actions.is_due(None,600,now))
        self.assertFalse(actions.is_due(now-timedelta(seconds=599),600,now))
        self.assertTrue(actions.is_due(now-timedelta(seconds=600),600,now))
        self.assertFalse(actions.is_due(now-timedelta(seconds=10799),10800,now))
        self.assertTrue(actions.is_due(now-timedelta(seconds=10800),10800,now))

    def test_cache_contains_only_market_data_and_can_restore_watermark(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(actions,'CACHE',Path(directory)):
            stamp=datetime.now(timezone.utc)
            t=np.array([int(stamp.timestamp()//60)-1,int(stamp.timestamp()//60)])
            prices=np.array([100.,101.])
            actions.cache_save(1,2,(t,prices),stamp)
            restored,watermark=actions.cache_load(1,2)
            np.testing.assert_array_equal(restored[0],t)
            np.testing.assert_array_equal(restored[1],prices)
            self.assertEqual(stamp,watermark)
            with np.load(Path(directory)/'1-2.npz',allow_pickle=False) as data:
                self.assertEqual(set(data.files),{'times','prices','watermark','project','stock'})

    def test_expired_or_corrupt_cache_causes_full_database_reload(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(actions,'CACHE',Path(directory)):
            stamp=datetime.now(timezone.utc)-timedelta(days=2)
            actions.cache_save(1,2,(np.array([100]),np.array([100.])),stamp)
            self.assertIsNone(actions.cache_load(1,2)[1])
            (Path(directory)/'1-2.npz').write_bytes(b'corrupt')
            self.assertIsNone(actions.cache_load(1,2)[1])

    def test_wrong_project_cannot_write_stock_outputs(self):
        conn=MagicMock()
        conn.__enter__.return_value=conn
        conn.execute.return_value.fetchone.side_effect=[(True,),(2,True)]
        with patch.object(actions,'connect',return_value=conn):
            with self.assertRaisesRegex(ValueError,'assignment mismatch'):
                actions.process(1,'project-one-only')
        statements=[call.args[0].lower() for call in conn.execute.call_args_list]
        self.assertTrue(all(query.lstrip().startswith('select') for query in statements))
        self.assertEqual(len(statements),2)

    def test_connection_diagnostics_never_expose_credentials(self):
        sensitive='postgresql://postgres:private-test-password@private-host/postgres'
        cases=[
            ('password authentication failed', 'authentication_failed'),
            ('Tenant or user not found', 'pooler_identity_invalid'),
            ('could not translate host name', 'hostname_not_found'),
            ('network is unreachable', 'network_unreachable'),
            ('SSL certificate verify failed', 'tls_connection_failed'),
            ('connection timeout expired', 'connection_timeout'),
        ]
        for message,expected in cases:
            details=safe_error(Exception(message+' '+sensitive),'connect')
            self.assertEqual(details['category'],expected)
            self.assertNotIn('private-test-password',str(details))
            self.assertNotIn('private-host',str(details))

    def test_hourly_cache_save_is_bounded_and_does_not_require_credentials(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(actions,'CACHE',Path(directory)):
            output=Path(directory)/'outputs.txt'
            with patch.dict(os.environ,{'GITHUB_OUTPUT':str(output)}):
                actions.cache_output()
                self.assertIn('cache_save=false',output.read_text())
                (Path(directory)/'1-2.npz').write_bytes(b'public market placeholder')
                actions.cache_output();actions.cache_output()
                self.assertEqual(output.read_text().splitlines()[-2:],['cache_save=true','cache_save=false'])


if __name__=='__main__':
    unittest.main()
