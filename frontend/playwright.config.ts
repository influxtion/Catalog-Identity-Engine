import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:3011",
    trace: "retain-on-failure",
    ...devices["Desktop Chrome"],
  },
  webServer: [
    {
      command: "node scripts/test-api.mjs",
      url: "http://127.0.0.1:8011/health",
      reuseExistingServer: false,
      timeout: 30000,
    },
    {
      command: "npm run start -- --port 3011",
      url: "http://127.0.0.1:3011",
      reuseExistingServer: false,
      timeout: 60000,
      env: { CATALOG_API_URL: "http://127.0.0.1:8011" },
    },
  ],
});
