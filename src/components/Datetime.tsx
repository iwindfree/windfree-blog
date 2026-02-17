import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import tz from "dayjs/plugin/timezone";
import { IconCalendar } from "./Icons";
import { SITE } from "@/config";

dayjs.extend(utc);
dayjs.extend(tz);

interface DatetimeProps {
  className?: string;
  size?: "sm" | "lg";
  pubDatetime: string | Date;
  timezone?: string;
  modDatetime?: string | Date | null;
}

export default function Datetime({
  pubDatetime,
  modDatetime,
  size = "sm",
  className = "",
  timezone: postTimezone,
}: DatetimeProps) {
  const isModified = modDatetime && modDatetime > pubDatetime;
  const datetime = dayjs(isModified ? modDatetime : pubDatetime).tz(
    postTimezone || SITE.timezone
  );
  const date = datetime.format("D MMM, YYYY");

  return (
    <div className={`flex items-center gap-x-2 opacity-80 ${className}`}>
      <IconCalendar
        className={`inline-block size-6 min-w-5.5 ${size === "sm" ? "scale-90" : ""}`}
      />
      {isModified && (
        <span className={`text-sm ${size === "lg" ? "sm:text-base" : ""}`}>
          Updated:
        </span>
      )}
      <time
        className={`text-sm ${size === "lg" ? "sm:text-base" : ""}`}
        dateTime={datetime.toISOString()}
      >
        {date}
      </time>
    </div>
  );
}
