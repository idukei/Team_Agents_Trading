from __future__ import annotations

import structlog
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType as AlpacaOrderType, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from core.config import settings

log = structlog.get_logger(__name__)

_client: TradingClient | None = None


def get_trading_client() -> TradingClient:
    global _client
    if _client is None:
        _client = TradingClient(
            api_key=settings.alpaca_api_key.get_secret_value(),
            secret_key=settings.alpaca_secret_key.get_secret_value(),
            paper=settings.alpaca_paper,
        )
        mode = "PAPER" if settings.alpaca_paper else "LIVE"
        log.info("alpaca_client_initialized", mode=mode)
    return _client


async def place_limit_order(
    symbol: str,
    side: str,                   # "buy" or "sell"
    qty: float,
    limit_price: float,
    time_in_force: str = "day",
) -> dict:
    """Place a limit order via Alpaca REST API."""
    client = get_trading_client()
    order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    tif = TimeInForce(time_in_force.lower()) if time_in_force.lower() in ("day", "gtc", "ioc", "fok") else TimeInForce.DAY

    request = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=order_side,
        type=AlpacaOrderType.LIMIT,
        time_in_force=tif,
        limit_price=limit_price,
    )
    order = client.submit_order(request)
    log.info("limit_order_placed", symbol=symbol, side=side, qty=qty, price=limit_price, order_id=str(order.id))
    return {
        "order_id": str(order.id),
        "status": str(order.status),
        "symbol": str(order.symbol),
        "qty": float(order.qty or qty),
        "limit_price": limit_price,
    }


async def place_market_order(
    symbol: str,
    side: str,
    qty: float,
) -> dict:
    """Place a market order via Alpaca REST API."""
    client = get_trading_client()
    order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=order_side,
        type=AlpacaOrderType.MARKET,
        time_in_force=TimeInForce.DAY,
    )
    order = client.submit_order(request)
    log.info("market_order_placed", symbol=symbol, side=side, qty=qty, order_id=str(order.id))
    return {
        "order_id": str(order.id),
        "status": str(order.status),
        "symbol": str(order.symbol),
        "qty": float(order.qty or qty),
    }


async def cancel_order(order_id: str) -> bool:
    """Cancel an open order."""
    try:
        client = get_trading_client()
        client.cancel_order_by_id(order_id)
        log.info("order_cancelled", order_id=order_id)
        return True
    except Exception as e:
        log.warning("order_cancel_failed", order_id=order_id, error=str(e))
        return False


async def get_order_status(order_id: str) -> dict | None:
    """Get current status of an order."""
    try:
        client = get_trading_client()
        order = client.get_order_by_id(order_id)
        return {
            "order_id": str(order.id),
            "status": str(order.status),
            "filled_qty": float(order.filled_qty or 0),
            "filled_avg_price": float(order.filled_avg_price or 0),
        }
    except Exception as e:
        log.warning("get_order_failed", order_id=order_id, error=str(e))
        return None


async def get_account() -> dict:
    client = get_trading_client()
    account = client.get_account()
    return {
        "cash": float(account.cash),
        "portfolio_value": float(account.portfolio_value),
        "buying_power": float(account.buying_power),
        "equity": float(account.equity),
    }
