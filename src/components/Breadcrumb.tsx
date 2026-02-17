"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Breadcrumb() {
  const pathname = usePathname();
  const currentUrlPath = pathname.replace(/\/+$/, "");
  const breadcrumbList = currentUrlPath.split("/").slice(1);

  if (breadcrumbList[0] === "blog" && breadcrumbList.length === 1) {
    breadcrumbList.splice(0, 1, "Blog");
  } else if (breadcrumbList[0] === "blog") {
    breadcrumbList.splice(0, 2, `Blog (page ${breadcrumbList[1] || 1})`);
  }

  if (breadcrumbList[0] === "tags" && !isNaN(Number(breadcrumbList[2]))) {
    breadcrumbList.splice(
      1,
      3,
      `${breadcrumbList[1]} ${Number(breadcrumbList[2]) === 1 ? "" : "(page " + breadcrumbList[2] + ")"}`
    );
  }

  return (
    <nav className="app-layout mt-8 mb-1" aria-label="breadcrumb">
      <ul className="font-light [&>li]:inline [&>li:not(:last-child)>a]:hover:opacity-100">
        <li>
          <Link href="/" className="opacity-80">
            Home
          </Link>
          <span aria-hidden="true" className="opacity-80">
            &raquo;
          </span>
        </li>
        {breadcrumbList.map((breadcrumb, index) =>
          index + 1 === breadcrumbList.length ? (
            <li key={breadcrumb}>
              <span
                className={`capitalize opacity-75 ${index > 0 ? "lowercase" : ""}`}
                aria-current="page"
              >
                {decodeURIComponent(breadcrumb)}
              </span>
            </li>
          ) : (
            <li key={breadcrumb}>
              <Link
                href={`/${breadcrumb}/`}
                className="capitalize opacity-70"
              >
                {breadcrumb}
              </Link>
              <span aria-hidden="true">&raquo;</span>
            </li>
          )
        )}
      </ul>
    </nav>
  );
}
