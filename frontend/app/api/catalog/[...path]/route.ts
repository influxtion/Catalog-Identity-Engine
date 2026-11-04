import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

async function proxy(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  const { path } = await context.params;
  const base = process.env.CATALOG_API_URL || "http://127.0.0.1:8000";
  const target = `${base}/${path.map(encodeURIComponent).join("/")}${request.nextUrl.search}`;
  try {
    const response = await fetch(target, {
      method: request.method,
      headers: {
        "Content-Type": "application/json",
        ...(path[0] === "annotations" && request.headers.get("Authorization")
          ? { Authorization: request.headers.get("Authorization")! }
          : {}),
      },
      body: request.method === "POST" ? await request.text() : undefined,
      cache: "no-store",
      signal: AbortSignal.timeout(30000),
    });
    return new NextResponse(await response.text(), {
      status: response.status,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return NextResponse.json(
      {
        detail:
          "The catalog service is unavailable. Start the API and try again.",
      },
      { status: 503 },
    );
  }
}

export const GET = proxy;
export const POST = proxy;
