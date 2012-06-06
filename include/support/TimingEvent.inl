#include <stdexcept>
#include <iostream>

#include "TimingEvent.h"

namespace support {

/*** Timing Event ***/
/* ctor/dtor */
inline TimingEvent::TimingEvent() : 
    m_name("unknown"), m_start(), m_end(), 
    m_duration(), m_num_pauses(0), 
    m_isrunning(false), m_isend(false) { }

inline TimingEvent::TimingEvent(std::string name) : 
    m_name(name), m_start(), m_end(), 
    m_duration(), m_num_pauses(0), m_isrunning(false), m_isend(false) { }

/* getter/setter */
inline const std::string& TimingEvent::name() const { return m_name; }
inline const Time& TimingEvent::start_time() const { return m_start; }
inline const Time& TimingEvent::end_time() const { return m_end; }
inline const unsigned& TimingEvent::num_pauses() const { return m_num_pauses; }
inline Time TimingEvent::tot_dur() const { 
  return getElapsedTime(m_start, m_end);
}
inline Time TimingEvent::run_dur() const { return m_duration; }
inline Time TimingEvent::ave_dur() const {
  if (m_num_pauses == 0) return Time(0, 0);
  return devideTime(m_duration, m_num_pauses);
}

/* start() */
inline void TimingEvent::start() {
  if (m_isrunning == false) {
    m_duration.sec = 0;
    m_duration.usec = 0;
    m_start = readCurrTime();
    m_prev = m_start;
    m_isrunning = true;
  }
}

/* pause() */
inline void TimingEvent::pause() {
  if (m_isrunning == true) {
    Time curr_dur = getElapsedTime(m_prev, readCurrTime());
    m_duration = increaseTime(curr_dur, m_duration);
    m_num_pauses ++;
    m_isrunning = false;
  }
}

/* restart() */
inline void TimingEvent::restart() {
  if (m_isrunning == false) {
    m_prev = readCurrTime();
    m_isrunning = true;
  }
}

/* end() */
inline void TimingEvent::end() {
  if (m_isrunning == true) {
    Time curr_dur = getElapsedTime(m_prev, readCurrTime());
    m_duration = increaseTime(curr_dur, m_duration);
    m_num_pauses ++;
    m_isrunning = false;
  }
  if (not m_isend) {
    m_end = readCurrTime();
    m_isend = true;
  }
}

/*** TimingSys ***/
/* ctor/dtor */
//dtor delete all pointers
inline void TimingSys::init() {
  getSet();
}

inline void TimingSys::finalize() {
  for (std::map<std::string, TimingEvent*>::iterator it = getSet()->begin();
       it != getSet()->end(); ++it) {
    delete (*it).second;
  }
}

/* newEvent() */
inline void TimingSys::newEvent(const std::string& name) {
  if (getSet()->find(name) != getSet()->end()) return;
  (*getSet())[name] = new TimingEvent(name);
}

/* delEvent() */
inline void TimingSys::delEvent(const std::string& name) {
  delete (*getSet())[name];
  getSet()->erase(name);
}

/* operator[] */
inline TimingEvent& TimingSys::get(const std::string& name) {
  checkNameInTableOrDie(name);
  return (*(*getSet())[name]);
}

/* startEvent() */
inline void TimingSys::startEvent(const std::string& name) {
  checkNameInTableOrDie(name);
  (*getSet())[name]->start();
}

/* pauseEvent() */
inline void TimingSys::pauseEvent(const std::string& name) {
  checkNameInTableOrDie(name);
  (*getSet())[name]->pause();
}

/* restartEvent() */
inline void TimingSys::restartEvent(const std::string& name) {
  checkNameInTableOrDie(name);
  (*getSet())[name]->restart();
}

/* endEvent() */
inline void TimingSys::endEvent(const std::string& name) {
  checkNameInTableOrDie(name);
  (*getSet())[name]->end();
}

/* begin() */
inline TimingSys::iterator TimingSys::begin() {
  return iterator(getSet()->begin());
}

/* end() */
inline TimingSys::iterator TimingSys::end() {
  return iterator(getSet()->end());
}

/* getSet()
 * get a static set pointer
 */
inline std::map<std::string, TimingEvent*>* TimingSys::getSet() {
  static std::map<std::string, TimingEvent*> ret;
  return &ret;
}

/* checkNameInTableOrDie() */
inline void TimingSys::checkNameInTableOrDie(const std::string& name){
  if (getSet()->find(name) == getSet()->end()) {
    throw std::runtime_error(
        "TimingSys operator[] event name not in table: " + name);
  }
}

/*** TimingSys::iterator ***/
/* ctor/dtor */
inline TimingSys::iterator::iterator(
    const iterator& it) : m_it(it.m_it) { }

inline TimingSys::iterator::iterator(
    std::map<std::string, TimingEvent*>::iterator map_it) : m_it(map_it){ }

/* operator++() */
inline void TimingSys::iterator::operator++() {
  m_it ++;
}

inline void TimingSys::iterator::operator++(int) {
  m_it ++;
}

/* operator*() */
inline TimingEvent& TimingSys::iterator::operator*() {
  return *((*m_it).second);
}

/* operator->() */
inline TimingEvent* TimingSys::iterator::operator->() {
  return (*m_it).second;
}

/* operator==() */
inline bool 
TimingSys::iterator::operator==(const iterator& it) const {
  return m_it == it.m_it;
}

/* operator!=() */
inline bool 
TimingSys::iterator::operator!=(const iterator& it) const {
  return m_it != it.m_it;
}

} /* namespace support */
