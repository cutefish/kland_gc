#include "support/exception/Exception.h"
#include "support/exception/StdErrCategory.h"

namespace support {

inline void throwError(bool is_success, int errno) {
  if (is_success) {
    throw Exception(ErrorCode(errno, getErrorCategory<StdErrCategory>()));
  }
}

ThreadsAttr::ThreadsAttr() {
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
}

ThreadsAttr::~ThreadsAttr() {
  pthread_attr_destroy(&attr);
}

pthread_attr_t ThreadsAttr::get() {
  return &m_attr;
}

inline bool ThreadsAttr::setAffinity() {
  m_isAffinity = true;
}

ThreadPool::ThreadPool(int num_threads, ThreadsAttr& attr) 
    : m_num_threads(num_threads), m_wait(true), m_exit(false), m_func(NULL), 
    m_bar(num_threads + 1)
{
  for (int i = 0; i < num_threads; ++i) {
    m_threadArgs[i].tid = i;
    m_threadArgs[i].pool = this;

    int errno = pthread_create(&m_threads[i], attr.get(), loop, &m_threadArgs[i]);
    throwError(errno != 0, errno);
  }
}

ThreadPool::~ThreadPool() {
  for (int i = 0; i < num_threads; ++i) {
    void* ret;
    int errno = pthread_join(m_threads[i], &ret);
    throwError(errno != 0, errno);
  }
}

void ThreadPool::set(UserThreadFunc func, std::vector<void*> user_args) {
  m_func = func;
  for (int i = 0; i < num_threads; ++i) {
    m_userArgs[i] = user_args[i];
  }
}

void ThreadPool::execute() {
  //control thread enters the barrier, which will release all the working
  //threads.
  m_wait = false;
  m_bar.wait();
}

void ThreadPool::wait() {
  //control thread enters a barrier, waiting for the other working threads.
  //The working threads will then enter another barrier to wait.
  m_wait = true;
  m_bar.wait();
}

void ThreadPool::exit() {
  m_exit = true;
}

void* ThreadPool::loop(void* args) {
  ThreadArgs* thread_args = (ThreadArgs*)args;

  int tid = thread_args->tid;
  ThreadPool* pool = thread_args->pool;

  //enter barrier and wait for start
  pool->m_bar.wait();

  while (!pool->m_exit) {
    if (pool->m_wait) {
      //enter first barrier to release control thread
      pool->m_bar.wait();
      //enter second barrier to wait for control thread
      pool->m_bar.wait();
    }
    else {
      UserThreadFunc func = pool->m_func;
      void* user_args = pool->user_args[tid];
      (*func)(pool, tid, user_args);
    }
  }

  pthread_exit(args);
}


} /* namespace support */
