CXX = icpx
CXXFLAGS = -g -O2 -fsycl

TARGETS = demo_1 demo_2 demo_3

all: $(TARGETS)

demo_1: demo_1.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

demo_2: demo_2.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

demo_3: demo_3.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean