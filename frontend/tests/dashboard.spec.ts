import { expect, test } from "@playwright/test";

test("explore, compare, and record a review", async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Identity explorer", exact: true }),
  ).toBeVisible();
  await expect(page.getByText("Synthetic showcase")).toBeVisible();
  await page.getByRole("textbox", { name: "Search listings" }).fill("Lumen");
  await expect(page.locator(".listing-table tbody tr")).toHaveCount(3);
  await page.getByRole("textbox", { name: "Search listings" }).clear();
  await page
    .getByRole("button", {
      name: "Select Aster H1 Wireless Headphones · Black",
      exact: true,
    })
    .click();
  await page
    .getByRole("button", { name: /Compare with Aster H-1 Noise/ })
    .click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(
    page.getByText("Same exact variant", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("textbox", { name: "Review note" })
    .fill("Verified matching model and black color.");
  await page
    .getByRole("button", { name: "Confirm variant match", exact: true })
    .click();
  await expect(
    page.getByText("Review saved. Identity updates are queued for the worker."),
  ).toBeVisible();
  await page.getByRole("button", { name: "Close comparison" }).click();
  await page
    .getByRole("button", { name: "Identity graph", exact: true })
    .click();
  await expect(
    page.getByRole("img", { name: "Identity relationship graph" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Evaluations", exact: true }).click();
  await expect(
    page.getByText("Precision target not met", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Enriched logistic", { exact: true }),
  ).toBeVisible();
});

test("invalid imports show an error and valid records persist", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Add listings", exact: true }).click();
  await page
    .getByRole("textbox", { name: "Listing JSON", exact: true })
    .fill("not valid json");
  await page
    .getByRole("button", { name: "Import listings", exact: true })
    .click();
  await expect(page.getByRole("dialog").getByRole("alert")).toBeVisible();
  await page
    .getByRole("textbox", { name: "Listing JSON", exact: true })
    .fill(
      JSON.stringify([
        {
          source: "test-source",
          source_id: "ui-import",
          title: "Fictional imported lamp",
        },
      ]),
    );
  await page
    .getByRole("button", { name: "Import listings", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page
    .getByRole("textbox", { name: "Search listings" })
    .fill("imported lamp");
  await expect(page.locator(".listing-table tbody tr")).toHaveCount(1);
});

test("mobile navigation and desktop screenshot", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto("/");
  await expect(page.locator(".listing-table tbody tr").first()).toBeVisible();
  await page.screenshot({
    path: "test-results/dashboard-desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(
    page.getByRole("heading", { name: "Identity explorer", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: /Review queue/ }).click();
  await expect(
    page.getByRole("heading", {
      name: "Pairs awaiting a decision",
      exact: true,
    }),
  ).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth,
  );
  expect(overflow).toBe(false);
  await page.screenshot({
    path: "test-results/dashboard-mobile.png",
    fullPage: true,
  });
});
