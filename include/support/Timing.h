#ifndef SUPPORT_TIMING_H_
#define SUPPORT_TIMING_H_

#include <string>

/*!\file Timing.h
 * \brief A collection of timer utilities.
 */

namespace support {

/* \struct Time
 * mirror the struct timeval
 */
struct Time {
  long sec;
  long usec;
  Time();
  Time(long, long);
};

/* readCurrTime()
 * Return the current time since Epoch.
 * The time is returned in the form of a long integer pair. The first element
 * represents the number of whole seconds of elapsed time. The second element
 * represents the number of microseconds.
 */
Time readCurrTime();

/* getElapsedTime()
 * Return time difference between start and end
 */
Time getElapsedTime(Time start, Time end, bool& is_negative);
Time getElapsedTime(const Time& start, const Time& end);

/* increaseTime()
 * Add two time object and return the result
 */
Time increaseTime(const Time& t0, const Time& t1);

/* devideTime()
 * Devide time by a integer and return result
 */
Time devideTime(const Time& duration, const unsigned& num);

/* Time2String()
 * Return a time string according to formatter string.
 * In the formatter string, special place holders will be replaced.
 * %EY --  year from epoch
 * %EM --  month from epoch
 * %ED --  day from epoch
 * %EH --  hour from epoch
 * %Em --  miniute from epoch
 * %Es --  second from epoch
 * %PY --  year part
 * %PM --  month part
 * %PD --  day part
 * %PH --  hour part
 * %Pm --  minute part
 * %Ps --  second part
 * %Pi --  milli-second part
 * %Pu --  micro-second part
 * %Pn --  nano-second part
 * %TY --  year total
 * %TM --  month total
 * %TD --  day total
 * %TH --  hour total
 * %Tm --  minute total
 * %Ts --  second total
 * %Ti --  milli-second total
 * %Tu --  micro-second total
 * %Tn --  nano-second total
 */
std::string Time2String(const Time& time, 
                        std::string formatter="%EY/%EM/%ED/%EH:%Em:%Es");


} /* namespace support */

#include "Timing.inl"
#endif /* SUPPORT_TIMING_H_ */
