import type { Metadata } from "next";
import { Suspense } from "react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import Breadcrumb from "@/components/Breadcrumb";
import SearchClient from "@/components/SearchClient";

export const metadata: Metadata = {
  title: "Search",
};

export default function SearchPage() {
  return (
    <>
      <link rel="stylesheet" href="/pagefind/pagefind-ui.css" />
      <Header />
      <Breadcrumb />
      <main id="main-content" className="app-layout pb-4">
        <h1 className="text-2xl font-semibold sm:text-3xl">Search</h1>
        <p className="mt-2 mb-6 italic">Search any article ...</p>
        <Suspense fallback={<div>Loading search...</div>}>
          <SearchClient />
        </Suspense>
      </main>
      <Footer />
    </>
  );
}
