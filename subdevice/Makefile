CXX = icpx
CXXFLAGS = -g -O2 -fsycl

TARGETS = demo-1 demo-2 demo-3

all: $(TARGETS)

demo-1: demo-1.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

demo-2: demo-2.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

demo-3: demo-3.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean