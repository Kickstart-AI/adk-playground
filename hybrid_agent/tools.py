"""Tool implementations and argument schemas for the hybrid agent."""

from pydantic import BaseModel, Field


class ToolArgs(BaseModel):
    """Common structured output for tool argument extraction."""

    transfer_to_agent: str = Field(
        "",
        description=(
            "Workflow target to route to instead of extracting arguments, currently intake."
        ),
    )


class FetchOrderArgs(ToolArgs):
    """Arguments for fetch_order."""

    order_number: str


class ValidateArgs(ToolArgs):
    """Arguments for validate_order_eligibility."""

    order_number: str


class RegisterReturnArgs(ToolArgs):
    """Arguments for register_return."""

    order_number: str
    item: str


class SendLoginNumberArgs(ToolArgs):
    """Arguments for send_login_number."""

    email: str


class ValidateLoginNumberArgs(ToolArgs):
    """Arguments for validate_login_number."""

    email: str
    login_number: str


class ChangePaymentMethodArgs(ToolArgs):
    """Arguments for change_payment_method."""

    email: str
    payment_method: str


class ChangeAddressArgs(ToolArgs):
    """Arguments for change_address."""

    email: str
    address: str


FAKE_ORDERS = {
    "1001": {
        "order_number": "1001",
        "items": [
            {"name": "running shoes", "returnable": True},
            {"name": "socks", "returnable": True},
            {"name": "protein bars", "returnable": False, "reason": "food is exempt from return"},
        ],
        "delivered_days_ago": 5,
    },
    "1002": {
        "order_number": "1002",
        "items": [{"name": "winter jacket", "returnable": True}],
        "delivered_days_ago": 45,
    },
}

FAKE_ACCOUNTS = {
    "casey@example.com": {
        "email": "casey@example.com",
        "login_number": "123456",
        "payment_method": "paypal",
        "address": "12 Market Street, Amsterdam",
    }
}


def fetch_order(order_number: str) -> dict:
    """Fetch order details by order number."""
    order = FAKE_ORDERS.get(order_number)
    if order is None:
        raise ValueError(f"Order {order_number} not found.")
    return order


def validate_order_eligibility(order_number: str) -> dict:
    """Check that the order is within the 30-day return window."""
    order = fetch_order(order_number)
    if order["delivered_days_ago"] > 30:
        raise ValueError(f"Order {order_number} is outside the 30-day return window.")
    return {"eligible": True}


def register_return(order_number: str, item: str) -> dict:
    """Register the return of an item and issue a return label."""
    match = next((i for i in fetch_order(order_number)["items"] if i["name"] == item), None)
    if match is None:
        raise ValueError(f"Item '{item}' is not part of order {order_number}.")
    if not match["returnable"]:
        raise ValueError(f"Item '{item}' is not returnable: {match['reason']}.")
    return {
        "return_id": f"R-{order_number}",
        "label_url": f"https://returns.example.com/R-{order_number}.pdf",
    }


def send_login_number(email: str) -> dict:
    """Send a login number to the account email address."""
    account = FAKE_ACCOUNTS.get(email)
    if account is None:
        raise ValueError(f"No account found for {email}.")
    return {"email": email, "sent": True}


def validate_login_number(email: str, login_number: str) -> dict:
    """Validate the login number sent to the account email address."""
    account = FAKE_ACCOUNTS.get(email)
    if account is None:
        raise ValueError(f"No account found for {email}.")
    if account["login_number"] != login_number:
        raise ValueError("The login number is not valid.")
    return {
        "email": email,
        "payment_method": account["payment_method"],
        "address": account["address"],
    }


def change_payment_method(email: str, payment_method: str) -> dict:
    """Update the account payment method."""
    account = FAKE_ACCOUNTS.get(email)
    if account is None:
        raise ValueError(f"No account found for {email}.")
    account["payment_method"] = payment_method
    return {"email": email, "payment_method": payment_method}


def change_address(email: str, address: str) -> dict:
    """Update the account address."""
    account = FAKE_ACCOUNTS.get(email)
    if account is None:
        raise ValueError(f"No account found for {email}.")
    account["address"] = address
    return {"email": email, "address": address}


TOOLS = {
    "fetch_order": (fetch_order, FetchOrderArgs),
    "validate_order_eligibility": (validate_order_eligibility, ValidateArgs),
    "register_return": (register_return, RegisterReturnArgs),
    "send_login_number": (send_login_number, SendLoginNumberArgs),
    "validate_login_number": (validate_login_number, ValidateLoginNumberArgs),
    "change_payment_method": (change_payment_method, ChangePaymentMethodArgs),
    "change_address": (change_address, ChangeAddressArgs),
}
