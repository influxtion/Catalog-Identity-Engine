import { Icon } from "@/components/icons";
import { Listing } from "@/lib/api";

export function ProductIcon({
  listing,
  large = false,
}: {
  listing: Listing;
  large?: boolean;
}) {
  return (
    <span
      className={`product-icon ${large ? "large" : ""} ${listing.data.category || ""}`}
    >
      <Icon name={listing.data.category || "grid"} size={large ? 44 : 24} />
    </span>
  );
}

export function Tags({ listing }: { listing: Listing }) {
  return (
    <span className="tags">
      {Object.entries(listing.data.attributes).map(
        ([key, value]) =>
          value && (
            <span key={key}>
              {key === "color" && (
                <i
                  className="color-dot"
                  style={{
                    background:
                      value === "silver"
                        ? "#bec4c2"
                        : value === "black"
                          ? "#303638"
                          : value === "blue"
                            ? "#517b9c"
                            : "#9ca69e",
                  }}
                />
              )}{" "}
              {value}
            </span>
          ),
      )}
    </span>
  );
}
