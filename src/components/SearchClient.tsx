"use client";

import { useEffect, useRef } from "react";
import { useSearchParams } from "next/navigation";

export default function SearchClient() {
  const containerRef = useRef<HTMLDivElement>(null);
  const searchParams = useSearchParams();
  const initializedRef = useRef(false);

  useEffect(() => {
    if (!containerRef.current || initializedRef.current) return;
    initializedRef.current = true;

    const container = containerRef.current;

    (async () => {
      if (process.env.NODE_ENV === "development") {
        container.innerHTML = `
          <div class="bg-muted/75 rounded p-4 space-y-4 mb-4">
            <p><strong>DEV mode Warning! </strong>You need to build the project at least once to see the search results during development.</p>
            <code class="block bg-black text-white px-2 py-1 rounded">pnpm run build</code>
          </div>
        `;
      }

      // @ts-expect-error — Missing types for @pagefind/default-ui package.
      const { PagefindUI } = await import("@pagefind/default-ui");

      const search = new PagefindUI({
        element: "#pagefind-search",
        showImages: false,
        showSubResults: true,
        processTerm: function (term: string) {
          const params = new URLSearchParams(window.location.search);
          params.set("q", term);
          history.replaceState(history.state, "", "?" + params.toString());
          return term;
        },
      });

      const query = searchParams.get("q");
      if (query) {
        search.triggerSearch(query);
      }

      const searchInput = document.querySelector(
        ".pagefind-ui__search-input"
      );
      const clearButton = document.querySelector(
        ".pagefind-ui__search-clear"
      );

      function resetSearchParam(e: Event) {
        if ((e.target as HTMLInputElement)?.value.trim() === "") {
          history.replaceState(history.state, "", window.location.pathname);
        }
      }

      searchInput?.addEventListener("input", resetSearchParam);
      clearButton?.addEventListener("click", resetSearchParam);
    })();
  }, [searchParams]);

  return <div id="pagefind-search" ref={containerRef} />;
}
