import os
import sys
from unittest.mock import MagicMock

# ۱. افزودن ریشه پروژه به مسیر پایتون
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared_objects.log import logger
from miner_objects.prop_net_order_placer import PropNetOrderPlacer
from vanta_api.miner_api_manager import MinerAPIManager

def start_local_miner():
    logger.info("=" * 65)
    logger.info("Starting Lightweight Local Vanta Miner (Sandbox Mode)")
    logger.info("=" * 65)

    # ۲. والت ماک
    mock_wallet = MagicMock()
    mock_wallet.hotkey.ss58_address = "5D4gJ9QfbcMg338813wz3MKuRofTKfE6zR3iPaGHaWEnNKoo"
    mock_wallet.name = "mock_wallet"
    mock_wallet.hotkey_str = "mock_hotkey"

    # ۳. متادیتا و نوتیفایر ساختگی
    mock_metagraph = MagicMock()
    mock_metagraph.hotkeys = [mock_wallet.hotkey.ss58_address]

    mock_slack = MagicMock()
    mock_slack.send_message = MagicMock()

    mock_config = MagicMock()
    mock_config.netuid = 8
    mock_config.api_host = '0.0.0.0'
    mock_config.api_rest_port = 8088

    # ۴. ماک Position Inspector جهت شبیه‌سازی ولیدیتور لوکال
    mock_validator = MagicMock()
    mock_validator.hotkey = "5MockValidatorHotkeyForLocalSandboxTesting111111"
    mock_validator.ip = "127.0.0.1"
    mock_validator.port = 8090

    mock_position_inspector = MagicMock()
    mock_position_inspector.get_possible_validators.return_value = [mock_validator]

    # ۵. ساخت Order Placer با ولیدیتور ماک
    order_placer = PropNetOrderPlacer(
        wallet=mock_wallet,
        metagraph_client=mock_metagraph,
        config=mock_config,
        is_testnet=False,
        position_inspector=mock_position_inspector,
        slack_notifier=mock_slack,
        running_unit_tests=True
    )

    # ۶. استارت سرور REST API روی پورت 8088
    api_manager = MinerAPIManager(
        prop_net_order_placer=order_placer,
        miner_hotkey=mock_wallet.hotkey.ss58_address,
        api_host=mock_config.api_host,
        api_rest_port=mock_config.api_rest_port,
        slack_notifier=mock_slack
    )

    logger.info(f"Miner REST API Server running natively on http://127.0.0.1:{mock_config.api_rest_port}")
    logger.info("Ready to accept live orders from mining/run_ml_strategy.py...")
    
    api_manager.run()

if __name__ == "__main__":
    try:
        start_local_miner()
    except KeyboardInterrupt:
        logger.info("Miner local test stopped.")