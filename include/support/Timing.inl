#include <sys/time.h>
#include <ctime>

#include <iostream>

#include "StringUtils.h"
#include "Timing.h"
#include "Type.h"

namespace support {

static const int EpochYear = 1900;

/* Time ctor */
inline Time::Time() : sec(0), usec(0) { }

inline Time::Time(long s, long us) : sec(s), usec(us) { }

/* readCurrTime() */
inline Time readCurrTime() {
  struct timeval curr;
  gettimeofday(&curr, NULL);
  Time ret;
  ret.sec = curr.tv_sec;
  ret.usec = curr.tv_usec;
  return ret;
}

/* getElapsedTime() */
inline Time getElapsedTime(Time start, Time end, bool& is_negative) {
  Time result;
  // Perform the carry for the later subtraction by updating start
  if (end.usec < start.usec) {
    int nsec = (start.usec - end.usec) / 1000000 + 1;
    start.usec -= 1000000 * nsec;
    start.sec += nsec;
  }
  if (end.usec - start.usec > 1000000) {
    int nsec = (end.usec - start.usec) / 1000000;
    start.usec += 1000000 * nsec;
    start.sec -= nsec;
  }

  // Compute the time remaining to wait.
  // usec is certainlstart positive.
  result.sec = end.sec - start.sec;
  result.usec = end.usec - start.usec;

  /* Return 1 if result is negative. */
  is_negative = end.sec < start.sec;

  return result;
}

inline Time getElapsedTime(const Time& start, const Time& end) {
  bool is_negative;
  return getElapsedTime(start, end, is_negative);
}

/* devideTime() */
inline Time devideTime(const Time& duration, const unsigned& num) {
  Time ret;
  ret.sec = duration.sec / num;
  ret.usec = ((duration.sec % num) * 1000000 + duration.usec) / num;
  return ret;
}

/* increaseTime() */
inline Time increaseTime(const Time& t0, const Time& t1) {
  Time ret;
  ret.usec = t0.usec + t1.usec;
  unsigned carry = ret.usec / 1000000;
  ret.usec = ret.usec - carry * 1000000;
  ret.sec = t0.sec + t1.sec + carry;
  return ret;
}

/* Time2String() */
inline std::string Time2String(const Time& time, std::string formatter) {
  struct tm* timeinfo;
  time_t seconds = static_cast<time_t>(time.sec);
  timeinfo = localtime(&seconds);
  replaceString(formatter, "%EY", Type2String<int>(timeinfo->tm_year + EpochYear));
  replaceString(formatter, "%EM", Type2String<int>(timeinfo->tm_mon + 1));
  replaceString(formatter, "%ED", Type2String<int>(timeinfo->tm_mday));
  replaceString(formatter, "%EH", Type2String<int>(timeinfo->tm_hour));
  replaceString(formatter, "%Em", Type2String<int>(timeinfo->tm_min));
  replaceString(formatter, "%Es", Type2String<int>(timeinfo->tm_sec));

  long year_offset = 3600 * 24 * 365;
  long month_offset = 3600 * 24 * 30;
  long day_offset = 3600 * 24;
  long hour_offset = 3600;
  long minute_offset = 60;
  replaceString(formatter, "%PY", Type2String<int>(seconds / year_offset));
  replaceString(formatter, "%PM", Type2String<int>(seconds % 
                                                   year_offset / month_offset));
  replaceString(formatter, "%PD", Type2String<int>(seconds %
                                                   month_offset / day_offset));
  replaceString(formatter, "%PH", Type2String<int>(seconds %
                                                   day_offset / hour_offset));
  replaceString(formatter, "%Pm", Type2String<int>(seconds %
                                                   hour_offset / minute_offset));
  replaceString(formatter, "%Ps", Type2String<int>(seconds % minute_offset));
  replaceString(formatter, "%Pi", Type2String<int>(time.usec/1000));
  replaceString(formatter, "%Pu", Type2String<int>(time.usec));
  replaceString(formatter, "%Pn", Type2String<int>(time.usec*1000));

  replaceString(formatter, "%TY", Type2String<int>(seconds / year_offset));
  replaceString(formatter, "%TM", Type2String<int>(seconds / month_offset));
  replaceString(formatter, "%TD", Type2String<int>(seconds / day_offset));
  replaceString(formatter, "%TH", Type2String<int>(seconds / hour_offset));
  replaceString(formatter, "%Tm", Type2String<int>(seconds / minute_offset));
  replaceString(formatter, "%Ts", Type2String<int>(seconds));
  replaceString(formatter, "%Ti", Type2String<int>(seconds * 1000 + 
                                                   time.usec/1000));
  replaceString(formatter, "%Tu", Type2String<int>(seconds * 1000000 + 
                                                   time.usec));
  replaceString(formatter, "%Tn", Type2String<int>(seconds * 1000000000 + 
                                                   time.usec*1000));

  return formatter;
}

} /* namespace support */
