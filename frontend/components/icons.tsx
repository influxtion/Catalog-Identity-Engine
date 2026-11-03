export function Icon({ name, size = 20 }: { name: string; size?: number }) {
  const paths: Record<string, React.ReactNode> = {
    search: (
      <>
        <circle cx="10.5" cy="10.5" r="6.5" />
        <path d="m16 16 5 5" />
      </>
    ),
    grid: (
      <>
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" />
      </>
    ),
    review: (
      <>
        <path d="M8 4H5v17h14V4h-3" />
        <rect x="8" y="2" width="8" height="5" rx="1" />
        <path d="m8 14 3 3 5-6" />
      </>
    ),
    graph: (
      <>
        <circle cx="5" cy="6" r="3" />
        <circle cx="19" cy="7" r="3" />
        <circle cx="12" cy="19" r="3" />
        <path d="m8 6 8 1M6 9l4 7m8-6-4 6" />
      </>
    ),
    chart: (
      <>
        <path d="M3 3v18h19M7 17v-5m5 5V6m5 11V9" />
      </>
    ),
    headphones: (
      <>
        <path d="M4 14v-3a8 8 0 0 1 16 0v3" />
        <rect x="3" y="12" width="5" height="9" rx="2" />
        <rect x="16" y="12" width="5" height="9" rx="2" />
      </>
    ),
    phone: (
      <>
        <rect x="6" y="2" width="12" height="20" rx="3" />
        <path d="M10 5h4m-3 14h2" />
      </>
    ),
    apparel: <path d="m8 3-6 4 3 5 3-2v11h8V10l3 2 3-5-6-4a4 4 0 0 1-8 0Z" />,
    arrow: <path d="M4 12h16m-6-6 6 6-6 6" />,
    close: <path d="m6 6 12 12M6 18 18 6" />,
    plus: <path d="M12 4v16M4 12h16" />,
    refresh: (
      <>
        <path d="M20 8a8 8 0 1 0 1 7M20 3v6h-6" />
      </>
    ),
    link: (
      <>
        <path d="m9 15 6-6M7 17l-1 1a4 4 0 0 1-5-5l4-4a4 4 0 0 1 6 0m2 6a4 4 0 0 0 6 0l4-4a4 4 0 0 0-5-5l-1 1" />
      </>
    ),
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name] || paths.grid}
    </svg>
  );
}
