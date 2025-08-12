from fastapi import APIRouter
from config import bed_collection, bed_capacity_collection
from models import BedSetup, Department

router = APIRouter(prefix="/beds", tags=["Beds"])

@router.post("/setup")
async def setup_beds(beds_setup: BedSetup):
    department = beds_setup.department.value
    new_total_beds = beds_setup.total_beds
    existing_count = await bed_collection.count_documents({"department": department})

    if new_total_beds <= existing_count:
        return {
            "message": f"{department} already has {existing_count} beds. No beds added."
        }

    beds_to_add = new_total_beds - existing_count
    prefix = {"ICU": "I", "Emergency": "E", "General Ward": "G"}[department]

    last_bed = await bed_collection.find({"department": department}).sort("bed_id", -1).limit(1).to_list(1)
    start_index = int(last_bed[0]["bed_id"].split("-")[1]) if last_bed else 0

    new_beds = [
        {"bed_id": f"{prefix}-{(start_index + i):03}", "department": department, "occupied": False}
        for i in range(1, beds_to_add + 1)
    ]
    if new_beds:
        await bed_collection.insert_many(new_beds)

    await bed_capacity_collection.update_one(
        {"_id": department},
        {"$set": {"total_beds": new_total_beds}},
        upsert=True
    )

    return {"message": f"Added {beds_to_add} new beds to {department}. Now total beds: {new_total_beds}"}

@router.get("/status")
async def bed_status():
    status = []
    for dept in Department:
        total = await bed_collection.count_documents({"department": dept.value})
        occupied = await bed_collection.count_documents({"department": dept.value, "occupied": True})
        available = total - occupied
        status.append({
            "department": dept.value,
            "total_beds": total,
            "available_beds": available,
            "occupied_beds": occupied
        })
    return status
