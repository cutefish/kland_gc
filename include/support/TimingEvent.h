#ifndef SUPPORT_TIMINGEVENT_H_
#define SUPPORT_TIMINGEVENT_H_

#include <map>
#include <string>

#include "Timing.h"

/*!\file TimingEvent.h
 * \brief This file expects to provide a easy to use timing interface. Timing
 * is done through operation on TimingEvent class objects such as start, pause
 * and end.
 */

namespace support {

/*!\class TimingEvent
 * \brief Timing event class. A timing event can start, pause, restart, and end.
 * The start time, number of pauses and duration of the event is automatically
 * recorded by the object. 
 */
class TimingEvent {
 public:
  /*** ctor/dtor ***/
  TimingEvent();
  TimingEvent(std::string name);

  /*** member function ***/
  /* getter/setter */
  const std::string& name() const;
  const Time& start_time() const;
  const Time& end_time() const;
  const unsigned& num_pauses() const;
  Time tot_dur() const;          //total duration (start - end)
  Time run_dur() const;          //running duration (not include pause time)
  Time ave_dur() const;          //average duration during start and pause

  /* start() */
  void start();

  /* pause() */
  void pause();

  /* restart() */
  void restart();

  /* end() */
  void end();

 private:
  std::string m_name;
  Time m_start;
  Time m_prev;
  Time m_end;
  Time m_duration;
  unsigned m_num_pauses;
  bool m_isrunning;
  bool m_isend;
};

/*!\class TimingSys
 * \brief static interface for easier access.
 */
class TimingSys {
 public:
  /*** ctor/dtor ***/
  static void init();
  static void finalize();

  /*** member function ***/
  /* newEvent() 
   * insert a new event.
   */
  static void newEvent(const std::string& name);

  /* delEvent() 
   * remove a event.
   */
  static void delEvent(const std::string& name);

  /* accessor for event 
   * if \p name does not match any event in the table, throw error
   */
  static TimingEvent& get(const std::string& name);

  /** event function wrappers
   * if \p name does not match any event in the table, throw error
   */
  /* startEvent() */
  static void startEvent(const std::string& name);

  /* pauseEvent() */
  static void pauseEvent(const std::string& name);

  /* restartEvent() */
  static void restartEvent(const std::string& name);

  /* endEvent() */
  static void endEvent(const std::string& name);

  /*** iterator ***/
  class iterator;
  friend class iterator;
  class iterator {
   public:
    /*** ctor/dtor ***/
    iterator(const iterator&);
    iterator(std::map<std::string, TimingEvent*>::iterator);
    
    /*** member function ***/
    void operator++ ();
    void operator++ (int);
    TimingEvent& operator* ();
    TimingEvent* operator-> ();
    bool operator==(const iterator&) const;
    bool operator!=(const iterator&) const;
   private:
    std::map<std::string, TimingEvent*>::iterator m_it;
  };

  static iterator begin();
  static iterator end();

 private:
  /*** private ctor ***/
  TimingSys();
  TimingSys(const TimingSys& other);
  TimingSys& operator= (const TimingSys& other);

  /*** helper funciton ***/
  static std::map<std::string, TimingEvent*>* getSet();
  static void checkNameInTableOrDie(const std::string& name);
};

} /* namespace support */

#include "TimingEvent.inl"

#endif /* SUPPORT_TIMINGEVENT_H_ */
