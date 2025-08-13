from fastapi import APIRouter, HTTPException
from config import user_collection, create_hash_password, verify_password, create_access_token, create_refresh_token
from models import SignupData, LoginData
from jose import jwt, JWTError
from config import SECRET_KEY, ALGORITHM

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.get("/")
def hello():
    return{"hello"}


@router.post("/signup")
async def signup(user: SignupData):
    if await user_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed_pass = create_hash_password(user.password)
    user_dict = user.model_dump()
    user_dict["hashed_password"] = hashed_pass
    del user_dict["password"]

    result = await user_collection.insert_one(user_dict)
    inserted_id = str(result.inserted_id)  # convert ObjectId to string for JSON

    return {
        "message": "Signup successful",
        "user": user.username,
        "id": inserted_id
    }

@router.post("/login")
async def login(user: LoginData):
    db_user = await user_collection.find_one({"username": user.username})
    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token({"sub": user.username})
    refresh_token = create_refresh_token({"sub": user.username})
    return {"access_token": access_token, "refresh_token": refresh_token, "role": db_user["role"]}


@router.post("/refresh")
async def refresh_token(refresh_token: str):
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=400, detail="Invalid token type")
        new_token = create_access_token({"sub": payload["sub"]})
        return {"access_token": new_token}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")


@router.get("/users")
async def get_all_users():
    users = await user_collection.find().to_list(1000)  # Fetch users

    if not users:
        raise HTTPException(status_code=404, detail="No users found")

    result = []
    for user in users:
        # Convert ObjectId to str
        if "_id" in user:
            user["_id"] = str(user["_id"])
        # Remove sensitive info
        user.pop("hashed_password", None)
        result.append(user)

    return {"users": result}
