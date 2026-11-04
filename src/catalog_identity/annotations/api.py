from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from catalog_identity.annotations.schema import SaveRequest
from catalog_identity.storage.catalog import Conflict, NotFound


def annotation_router(store):
    router = APIRouter(prefix="/annotations", tags=["annotations"])
    bearer = HTTPBearer(auto_error=False)
    credential_dependency = Depends(bearer)

    def actor(
        response: Response, credentials: HTTPAuthorizationCredentials | None = credential_dependency
    ):
        response.headers["Cache-Control"] = "no-store"
        member = store.authenticate(credentials.credentials) if credentials else None
        if not member:
            raise HTTPException(
                401,
                "A valid reviewer access key is required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return member

    def call(function, *args):
        try:
            return function(*args)
        except NotFound as exc:
            raise HTTPException(404, str(exc)) from exc
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    actor_dependency = Depends(actor)

    @router.get("/session")
    def session(member=actor_dependency):
        return member

    @router.get("/cases")
    def queue(member=actor_dependency):
        return call(store.queue, member)

    @router.get("/cases/{case_id}")
    def case(case_id: str, member=actor_dependency):
        return call(store.case, member, case_id)

    @router.post("/cases/{case_id}")
    def save(case_id: str, request: SaveRequest, member=actor_dependency):
        return call(store.save, member, case_id, request)

    return router
