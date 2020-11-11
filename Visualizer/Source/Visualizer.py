import argparse
import multiprocessing
import os
import shutil

from Visualizer.Plotting.Plot import Plot
from Visualizer.Testing.Persistence.TestSession import TestSession


def parse_arguments():
    parser = argparse.ArgumentParser(prog="Visualizer", description="Visualize EnergyManager test results.")

    parser.add_argument(
        "--update",
        "-u",
        action="store_true",
        help="Updates existing results (may take a long time).",
        required=False
    )

    input_group = parser.add_argument_group("input")
    input_group.add_argument(
        "--database",
        "-d",
        metavar="FILE",
        action="store",
        help="Specifies the database to use.",
        type=str,
        required=True
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--output-directory",
        "-o",
        metavar="DIRECTORY",
        action="store",
        help="Output directory to use for storing results.",
        type=str,
        required=True
    )

    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_arguments()

    # Load the tests
    tests = TestSession.load_all(arguments.database)

    # Set up statistics variable
    processed_tests = multiprocessing.Value("i", 0)


    def initialize(arguments):
        global processed_tests
        processed_tests = arguments


    def process_test(test_session: TestSession, test_count: int, output_directory: str, update: bool):
        global processed_tests

        with processed_tests.get_lock():
            processed_tests.value += 1

            print(f"Processing test {processed_tests.value}/{test_count}: {test_session.id} - {test_session.test_name}...")

        # Determine the output directory for the current test
        test_output_directory = f"{output_directory}/{test_session.id} - {test_session.test_name}"

        # Check if the results already exists
        if os.path.exists(test_output_directory):
            # If so, check if we need to update them
            if update:
                print(f"Results exist, updating...")

                # Delete the old results
                shutil.rmtree(test_output_directory)
            else:
                print(f"Results exist, skipping...")

                # Go to the next test
                return

        # Create a directory to hold the results
        os.makedirs(test_output_directory)

        # Plot some graphs
        overview_plot = test_session.profiler_session.overview_multi_plot()
        overview_plot.save(output_directory)

        # Free up memory between plots
        Plot.free()


    # Create a thread pool and process all threads
    with multiprocessing.Pool(initializer=initialize, initargs=(processed_tests,)) as pool:
        pool.starmap(process_test, [(test, len(tests), arguments.output_directory, arguments.update) for test in tests], chunksize=1)
