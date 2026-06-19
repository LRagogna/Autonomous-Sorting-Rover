Progress Log

May 24, 2026
- Defined project goals and requirements for the autonomous sorting rover
- Discussed object sorting methodology, hardware constraints, budget, expected capabilities
- Identified major subsystems --> mobility, computer vision, object retrieval
- Determined overall structure as follows
	- Vehicle chassis will consit of standard tank-built with wheels. Wheels on each side move together
	- Electromagnet will be used to pick up metal debris
	- Cameras will be used for object recognition (shape and color) 
	- Raspberry Pi 4 and Coral TPU will determine objection recognition behavior and rover decision making
- Starting sourcing parts

May 28, 2026
- Created final list of items to purchase
- Purchased items 
- Created inital sketches of project architecture

**waiting for items to arrive from amazon late may - early june**

June 14, 2026
- Begin breadboarding proof of concept for microcontroller powered electromagnet activation
	- Used relay module and LED light to represent behavior for activating electromagnet
	- Used Arduino in demo. Must decide between Arduino and ESP32 for actual implementation
- Finalize sketch of electrical layout for the project

June 15-16, 2026
- Received Raspberry Pi 4. Set up and gather other necessary parts to interact with Pi (keyboard, monitor, etc)
- Set up SSH for Raspberry pi and created basic programs, learning RP fundamentals

June 18, 2026
- Created basic object recognition program with RP camera (OV5647) and openCV
	- Verified camera detection with program --> rectangle_detect.py in /tests/
	- Program classifies green objects and highlights them in user view
- Cleaned up documentation and configured GitHub on RP

June 19, 2026
- Set up cooling system for RP
