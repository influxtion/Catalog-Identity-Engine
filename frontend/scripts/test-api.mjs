import { spawn, spawnSync } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const root = resolve("..");
const python =
  process.platform === "win32"
    ? join(root, ".venv", "Scripts", "python.exe")
    : join(root, ".venv", "bin", "python");
const folder = mkdtempSync(join(tmpdir(), "catalog-ui-"));
const env = {
  ...process.env,
  DATABASE_URL: `sqlite:///${join(folder, "catalog.db").replaceAll("\\", "/")}`,
  CATALOG_MODEL_DIR: "",
};
const seeded = spawnSync(
  python,
  ["-m", "catalog_identity.cli", "seed-showcase"],
  { cwd: root, env, stdio: "inherit" },
);
if (seeded.status !== 0) process.exit(seeded.status || 1);
const child = spawn(
  python,
  ["-m", "catalog_identity.cli", "serve", "--port", "8011"],
  { cwd: root, env, stdio: "inherit" },
);
for (const signal of ["SIGINT", "SIGTERM"])
  process.on(signal, () => child.kill());
child.on("exit", (code) => process.exit(code || 0));
