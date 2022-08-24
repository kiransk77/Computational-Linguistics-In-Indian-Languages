PYTHON = python3


# The @ makes sure that the command itself isn't echoed in the terminal
help:
	@echo "---------------HELP-----------------"
	@echo "To run the assignment type make run"
	@echo "------------------------------------"


# The ${} notation is specific to the make syntax and is very similar to bash's $() 	
run:

	@cd Q1
	@${PYTHON} Q1.py
	@cd ../Q2
	@${PYTHON} Q2.py
	@cd ../Q3
	@${PYTHON} Q3.py
	
	@echo all file generated
