import asyncio
import json
from datetime import datetime, timedelta
import random
from common.config import ARTIFICIAL_DELAY, MOCK_DATA_SIZE
import pathlib


def save_mock_data(data):
    """Save mock data to a timestamped file in mock_data_outputs directory."""
    # Create mock_data_outputs directory if it doesn't exist
    output_dir = pathlib.Path("mock_data_outputs")
    output_dir.mkdir(exist_ok=True)

    # Clean up old mock data files
    cleanup_mock_data_files(output_dir)

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"mock_data_{timestamp}.json"

    # Save the data with pretty printing
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nMock data saved to: {output_file}")


def cleanup_mock_data_files(output_dir):
    """Remove all existing mock data files in the output directory."""
    for file in output_dir.glob("mock_data_*.json"):
        try:
            file.unlink()
        except Exception as e:
            print(f"Warning: Could not delete {file}: {e}")


# Mock data generation
def generate_mock_data():
    customers = []
    appointments = []
    orders = []

    # Generate customers
    for i in range(MOCK_DATA_SIZE["customers"]):
        customer = {
            "id": f"CUST{i:04d}",
            "name": f"Customer {i}",
            "phone": f"+1555{i:07d}",
            "email": f"customer{i}@example.com",
            "joined_date": (
                datetime.now() - timedelta(days=random.randint(0, 7))
            ).isoformat(),
        }
        customers.append(customer)

    # Generate appointments
    for i in range(MOCK_DATA_SIZE["appointments"]):
        customer = random.choice(customers)
        appointment = {
            "id": f"APT{i:04d}",
            "customer_id": customer["id"],
            "customer_name": customer["name"],
            "date": (datetime.now() + timedelta(days=random.randint(0, 7))).isoformat(),
            "service": random.choice(
                ["Consultation", "Follow-up", "Review", "Planning"]
            ),
            "status": random.choice(["Scheduled", "Completed", "Cancelled"]),
        }
        appointments.append(appointment)

    # Generate orders
    for i in range(MOCK_DATA_SIZE["orders"]):
        customer = random.choice(customers)
        order = {
            "id": f"ORD{i:04d}",
            "customer_id": customer["id"],
            "customer_name": customer["name"],
            "date": (datetime.now() - timedelta(days=random.randint(0, 7))).isoformat(),
            "items": random.randint(1, 5),
            "total": round(random.uniform(10.0, 500.0), 2),
            "status": random.choice(["Pending", "Shipped", "Delivered", "Cancelled"]),
        }
        orders.append(order)

    # Format sample data for display
    sample_data = []
    sample_customers = random.sample(customers, 3)
    for customer in sample_customers:
        customer_data = {
            "Customer": customer["name"],
            "ID": customer["id"],
            "Phone": customer["phone"],
            "Email": customer["email"],
            "Appointments": [],
            "Orders": [],
        }

        # Add appointments
        customer_appointments = [
            a for a in appointments if a["customer_id"] == customer["id"]
        ]
        for apt in customer_appointments[:2]:
            customer_data["Appointments"].append(
                {
                    "Service": apt["service"],
                    "Date": apt["date"][:10],
                    "Status": apt["status"],
                }
            )

        # Add orders
        customer_orders = [o for o in orders if o["customer_id"] == customer["id"]]
        for order in customer_orders[:2]:
            customer_data["Orders"].append(
                {
                    "ID": order["id"],
                    "Total": f"${order['total']}",
                    "Status": order["status"],
                    "Date": order["date"][:10],
                    "# Items": order["items"],
                }
            )

        sample_data.append(customer_data)

    # Create data object
    mock_data = {
        "customers": customers,
        "appointments": appointments,
        "orders": orders,
        "sample_data": sample_data,
    }

    # Save the mock data
    save_mock_data(mock_data)

    return mock_data


# Initialize mock data
MOCK_DATA = generate_mock_data()


async def simulate_delay(delay_type):
    """Simulate processing delay based on operation type."""
    await asyncio.sleep(ARTIFICIAL_DELAY[delay_type])


async def get_customer(phone=None, email=None, customer_id=None):
    """Look up a customer by phone, email, or ID."""
    await simulate_delay("database")

    if phone:
        customer = next(
            (c for c in MOCK_DATA["customers"] if c["phone"] == phone), None
        )
    elif email:
        customer = next(
            (c for c in MOCK_DATA["customers"] if c["email"] == email), None
        )
    elif customer_id:
        customer = next(
            (c for c in MOCK_DATA["customers"] if c["id"] == customer_id), None
        )
    else:
        return {"error": "No search criteria provided"}

    return customer if customer else {"error": "Customer not found"}


async def get_customer_appointments(customer_id):
    """Get all appointments for a customer."""
    await simulate_delay("database")

    appointments = [
        a for a in MOCK_DATA["appointments"] if a["customer_id"] == customer_id
    ]
    return {"customer_id": customer_id, "appointments": appointments}


async def get_customer_orders(customer_id):
    """Get all orders for a customer."""
    await simulate_delay("database")

    orders = [o for o in MOCK_DATA["orders"] if o["customer_id"] == customer_id]
    return {"customer_id": customer_id, "orders": orders}


async def schedule_appointment(customer_id, date, service):
    """Schedule a new appointment."""
    await simulate_delay("database")

    # Verify customer exists
    customer = await get_customer(customer_id=customer_id)
    if "error" in customer:
        return customer

    # Create new appointment
    appointment_id = f"APT{len(MOCK_DATA['appointments']):04d}"
    appointment = {
        "id": appointment_id,
        "customer_id": customer_id,
        "customer_name": customer["name"],
        "date": date,
        "service": service,
        "status": "Scheduled",
    }

    MOCK_DATA["appointments"].append(appointment)
    return appointment


async def get_available_appointment_slots(start_date, end_date):
    """Get available appointment slots."""
    await simulate_delay("database")

    # Convert dates to datetime objects
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)

    # Generate available slots (9 AM to 5 PM, 1-hour slots)
    slots = []
    current = start
    while current <= end:
        if current.hour >= 9 and current.hour < 17:
            slot_time = current.isoformat()
            # Check if slot is already taken
            taken = any(a["date"] == slot_time for a in MOCK_DATA["appointments"])
            if not taken:
                slots.append(slot_time)
        current += timedelta(hours=1)

    return {"available_slots": slots}


async def prepare_agent_filler_message(websocket, message_type):
    """
    Handle agent filler messages while maintaining proper function call protocol.
    Returns a simple confirmation first, then sends the actual message to the client.
    """
    # First prepare the result that will be the function call response
    result = {"status": "queued", "message_type": message_type}

    # Prepare the inject message but don't send it yet
    if message_type == "lookup":
        inject_message = {
            "type": "InjectAgentMessage",
            "message": "Let me look that up for you...",
        }
    else:
        inject_message = {
            "type": "InjectAgentMessage",
            "message": "One moment please...",
        }

    # Return the result first - this becomes the function call response
    # The caller can then send the inject message after handling the function response
    return {"function_response": result, "inject_message": inject_message}


async def prepare_farewell_message(websocket, farewell_type):
    """End the conversation with an appropriate farewell message and close the connection."""
    # Prepare farewell message based on type
    if farewell_type == "thanks":
        message = "Thank you for calling! Have a great day!"
    elif farewell_type == "help":
        message = "I'm glad I could help! Have a wonderful day!"
    else:  # general
        message = "Goodbye! Have a nice day!"

    # Prepare messages but don't send them
    inject_message = {"type": "InjectAgentMessage", "message": message}

    close_message = {"type": "close"}

    # Return both messages to be sent in correct order by the caller
    return {
        "function_response": {"status": "closing", "message": message},
        "inject_message": inject_message,
        "close_message": close_message,
    }
