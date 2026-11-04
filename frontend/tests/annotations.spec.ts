import { expect, test, Page } from "@playwright/test";
import { readFileSync } from "node:fs";

function keyFor(name: string) {
  const access = JSON.parse(
    readFileSync("../runs/annotation-e2e/access.json", "utf-8"),
  );
  return access.members.find((member: { name: string }) => member.name === name)
    .access_key;
}

async function login(page: Page, name: string) {
  await page.getByLabel("Reviewer access key").fill(keyFor(name));
  await page.getByRole("button", { name: "Open review workspace" }).click();
  await expect(page.getByRole("button", { name: "Sign out" })).toBeVisible();
}

async function decision(page: Page, benchmark: string) {
  await page
    .getByLabel("Input sufficiency", { exact: true })
    .selectOption("sufficient");
  await page
    .getByLabel("Benchmark identity", { exact: true })
    .selectOption(benchmark);
  await page.getByLabel("Product family", { exact: true }).selectOption("same");
  await page
    .getByLabel("Exact variant", { exact: true })
    .selectOption("different");
  await page
    .getByRole("button", { name: "+ Add evidence", exact: true })
    .click();
  await page.getByLabel("Exact quote", { exact: true }).fill("blue");
  await page
    .getByLabel("Evidence role", { exact: true })
    .selectOption("variant_attribute");
  await page
    .getByLabel("What this supports", { exact: true })
    .fill("The listing explicitly describes a blue finish.");
  await page
    .getByLabel("Reasoning and remaining uncertainty", { exact: true })
    .fill("The colors differ. Benchmark identity remains a separate judgment.");
}

async function lock(page: Page) {
  await page
    .getByRole("button", { name: "Submit review…", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Submit and lock", exact: true })
    .click();
  await expect(page.getByRole("status")).toContainText("Submitted and locked");
  await expect(
    page.getByLabel("Benchmark identity", { exact: true }),
  ).toBeDisabled();
}

test("independent reviewers save, resume and lock before adjudication", async ({
  page,
}, testInfo) => {
  await page.goto("/annotate");
  await page.getByLabel("Reviewer access key").fill("invalid");
  await page.getByRole("button", { name: "Open review workspace" }).click();
  await expect(page.locator(".ann-error")).toContainText(
    "valid reviewer access key",
  );
  await login(page, "alice");
  await page
    .getByLabel("Reasoning and remaining uncertainty")
    .fill("A private initial note from reviewer Alice.");
  await page.getByRole("button", { name: "Save draft", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("Draft saved");
  await page.reload();
  await expect(
    page.getByLabel("Reasoning and remaining uncertainty"),
  ).toHaveValue("A private initial note from reviewer Alice.");
  await decision(page, "uncertain");
  await page.getByRole("button", { name: "+ Add source", exact: true }).click();
  await page
    .getByLabel("Source URL")
    .fill("https://example.org/fictional-lamp");
  await page.getByLabel("Date checked").fill("2026-01-01");
  await page
    .getByLabel("Finding and provenance")
    .fill("Fictional source entry used only to exercise the annotation form.");
  await page.getByRole("button", { name: "Save draft", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("Draft saved");
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: testInfo.outputPath("annotation-desktop.png"),
    fullPage: true,
  });
  await lock(page);
  await page.getByRole("button", { name: "Sign out" }).click();
  await login(page, "bob");
  await expect(
    page.getByLabel("Reasoning and remaining uncertainty"),
  ).toHaveValue("");
  await expect(page.getByLabel("Exact quote")).toHaveCount(0);
  await expect(
    page.getByText("A private initial note from reviewer Alice."),
  ).toHaveCount(0);
  await decision(page, "different");
  await lock(page);
  await page.getByRole("button", { name: "Sign out" }).click();
  await login(page, "carol");
  await expect(
    page.getByRole("heading", { name: "Independent reviews", exact: true }),
  ).toBeVisible();
  await expect(page.getByText("Disagreement", { exact: true })).toBeVisible();
  await decision(page, "uncertain");
  await lock(page);
  await page.getByLabel("Show cases").selectOption("submitted");
  await expect(page.locator(".ann-case-list button")).toHaveCount(1);
});

test("review workspace remains usable on a narrow screen", async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/annotate");
  await login(page, "alice");
  await expect(
    page.getByRole("heading", { name: "What do these listings support?" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: testInfo.outputPath("annotation-mobile.png"),
    fullPage: true,
  });
});

test("repeated codes require a precise quote location", async ({ page }) => {
  await page.goto("/annotate");
  await login(page, "alice");
  await page
    .getByRole("button", { name: "+ Add evidence", exact: true })
    .click();
  await page.getByLabel("Field", { exact: true }).selectOption("description");
  await page.getByLabel("Exact quote", { exact: true }).fill("R7");
  await page
    .getByLabel("Evidence role", { exact: true })
    .selectOption("compatibility");
  await page
    .getByLabel("What this supports")
    .fill("This occurrence refers to the compatible hub.");
  await page.getByRole("button", { name: "Save draft", exact: true }).click();
  await expect(page.locator(".ann-error")).toContainText(
    "quote occurs more than once",
  );
  await page
    .getByLabel("Quote location", { exact: true })
    .selectOption({ index: 2 });
  const location = await page
    .getByLabel("Quote location", { exact: true })
    .inputValue();
  await page.getByRole("button", { name: "Save draft", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("Draft saved");
  await page.reload();
  await expect(page.getByLabel("Quote location", { exact: true })).toHaveValue(
    location,
  );
});
