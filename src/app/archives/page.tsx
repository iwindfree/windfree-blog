import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { getAllPosts } from "@/lib/posts";
import getSortedPosts from "@/utils/getSortedPosts";
import getPostsByGroupCondition from "@/utils/getPostsByGroupCondition";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import Breadcrumb from "@/components/Breadcrumb";
import CardSimple from "@/components/CardSimple";
import { SITE } from "@/config";

export const metadata: Metadata = {
  title: "Archives",
};

const months = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

export default function ArchivesPage() {
  if (!SITE.showArchives) {
    redirect("/404");
  }

  const posts = getAllPosts().filter((p) => !p.data.draft);
  const sortedPosts = getSortedPosts(posts);

  return (
    <>
      <Header />
      <Breadcrumb />
      <main id="main-content" className="app-layout pb-4">
        <h1 className="text-2xl font-semibold sm:text-3xl">Archives</h1>
        <p className="mt-2 mb-6 italic">
          All the articles I&apos;ve archived.
        </p>
        {Object.entries(
          getPostsByGroupCondition(
            sortedPosts,
            (post) => post.data.pubDatetime.getFullYear()
          )
        )
          .sort(([yearA], [yearB]) => Number(yearB) - Number(yearA))
          .map(([year, yearGroup]) => (
            <div key={year}>
              <span className="text-2xl font-bold">{year}</span>
              <sup className="text-sm">{yearGroup.length}</sup>
              {Object.entries(
                getPostsByGroupCondition(
                  yearGroup,
                  (post) => post.data.pubDatetime.getMonth() + 1
                )
              )
                .sort(
                  ([monthA], [monthB]) => Number(monthB) - Number(monthA)
                )
                .map(([month, monthGroup]) => (
                  <div key={month} className="flex flex-col sm:flex-row">
                    <div className="mt-6 min-w-36 text-lg sm:my-6">
                      <span className="font-bold">
                        {months[Number(month) - 1]}
                      </span>
                      <sup className="text-xs">{monthGroup.length}</sup>
                    </div>
                    <ul>
                      {monthGroup
                        .sort(
                          (a, b) =>
                            Math.floor(
                              new Date(b.data.pubDatetime).getTime() / 1000
                            ) -
                            Math.floor(
                              new Date(a.data.pubDatetime).getTime() / 1000
                            )
                        )
                        .map((post) => (
                          <CardSimple key={post.id} post={post} />
                        ))}
                    </ul>
                  </div>
                ))}
            </div>
          ))}
      </main>
      <Footer />
    </>
  );
}
