
    # Import csv
import csv
# Import os
import os
import shutil
# Main Function
def main():
    # Open dataset file
    dataset = open('trainLabels.csv', newline='')

    # Initialize csvreader for dataset
    reader = csv.reader(dataset)

    # Read data from reader
    data = list(reader)

    # Variables for progress counter
    lines = len(data)
    i = 0

    # Analyze data in dataset
    for row in data:
        # Assign image name and state to variables
        image = row[0] + '.jpeg'
        state = row[1]

        # Print image information
        print('({}/{}) Processing image ({}): {}'.format(i + 1, lines, state, image))

        # Increment i
        i += 1
        # Determine action to perform
        if state == '0':
           # print ("human")
            # Attempt to move the file
            try:
                # Move the file to folderone/
                shutil.copy(image, 'notdiseased/' + image)
                # Inform the user of action being taken
                print(' -> Moved to not diseased/')
            except FileNotFoundError:
                # Inform the user of the failure
                print(' -> Failed to find file')
        elif state in ['1', '2', '3', '4']:
            # Attempt to move the file
            try:
                # Move the file to foldertwo/
                shutil.copy(image, 'diseased/' + image)
                # Inform the user of action being taken
                print(' -> Moved to diseased/')
            except FileNotFoundError:
                # Inform the user of the failure
                print(' -> Failed to find file')

# Execute main function if name is equal to main
if __name__ == '__main__':
    main()
