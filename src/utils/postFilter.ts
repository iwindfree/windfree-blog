import type { BlogPost } from "@/lib/posts";
import { SITE } from "@/config";

const postFilter = ({ data }: BlogPost) => {
  const isPublishTimePassed =
    Date.now() >
    new Date(data.pubDatetime).getTime() - SITE.scheduledPostMargin;
  return (
    !data.draft &&
    (process.env.NODE_ENV === "development" || isPublishTimePassed)
  );
};

export default postFilter;
