import logging
import os
import uuid
import subprocess
from pathlib import Path
from typing import List

from ..agents import (
    DescriptionAnalyzerAgent,
    ProfilingAgent,
    GuidelineAgent,
    PreprocessingCoderAgent,
    ModelingCoderAgent,
    AssemblerAgent,
)
from ..llm import ChatLLMFactory

# Basic configuration
logging.basicConfig(level=logging.INFO)

# Create a logger
logger = logging.getLogger(__name__)


class Manager:
    def __init__(
        self,
        input_data_folder: str,
        output_folder: str,
        config: str,
    ):
        """Initialize Manager with required paths and config from YAML file.

        Args:
            input_data_folder: Path to input data directory
            output_folder: Path to output directory
            config_path: Path to YAML configuration file
        """
        self.input_data_folder = input_data_folder
        self.output_folder = output_folder
        self.config = config

        # Validate paths
        for path, name in [(input_data_folder, "input_data_folder")]:
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} not found: {path}")

        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        self.description_analyzer_agent = DescriptionAnalyzerAgent(
            config=config,
            manager=self,
            llm_config=self.config.description_analyzer,
        )
        self.profiling_agent = ProfilingAgent(
            config=config,
            manager=self,
        )
        self.guideline_agent = GuidelineAgent(
            config=config,
            manager=self,
            llm_config=self.config.guideline_generator,
        )
        self.preprocessing_coder_agent = PreprocessingCoderAgent(
            config=config,
            manager=self,
            llm_config=self.config.preprocessing_coder,
        )
        self.modeling_coder_agent = ModelingCoderAgent(
            config=config,
            manager=self,
            llm_config=self.config.modeling_coder,
        )
        self.assembler_agent = AssemblerAgent(
            config=config,
            manager=self,
            llm_config=self.config.assembler,
        )

        self.context = {
            "input_data_folder": input_data_folder,
            "output_folder": output_folder,
            
        }

    def run_pipeline_new(self):
        """Run the entire pipeline from description analysis to code generation."""
        logger.info("Starting AutoML pipeline...")

        


        logger.info("AutoML pipeline completed successfully!")

    def run_pipeline(self):
        """Run the entire pipeline from description analysis to code generation."""

        # Step 1: Run description analysis agent
        analysis_result = self.description_analyzer_agent()
        if "error" in analysis_result:
            logger.error(f"Description analysis failed: {analysis_result['error']}")
            return
        logger.info(f"Analysis result: {analysis_result}")

        self.description_analysis = analysis_result

        # Step 2: Run profiling agent
        profiling_result = self.profiling_agent()
        if "error" in profiling_result:
            logger.error(f"Data profiling failed: {profiling_result['error']}")
            return
        
        self.profiling_result = profiling_result
        logger.info("Profiling overview generated.")

        # Step 3: Run guideline agent
        guideline = self.guideline_agent()
        if "error" in guideline:
            logger.error(f"Guideline generation failed: {guideline['error']}")
            return
        
        self.guideline = guideline
        logger.info("Guideline generated successfully.")

        # Step 4: Run Preprocessing Coder Agent
        preprocessing_code_result = self.preprocessing_coder_agent()
        if preprocessing_code_result.get("status") == "failed":
            logger.error(f"Preprocessing code generation failed: {preprocessing_code_result.get('error')}")
            return

        self.preprocessing_code = preprocessing_code_result.get("code")
        logger.info("Preprocessing code generated and validated successfully.")

        # Step 5: Run Modeling Coder Agent
        modeling_code_result = self.modeling_coder_agent()
        if modeling_code_result.get("status") == "failed":
            logger.error(f"Modeling code generation failed: {modeling_code_result.get('error')}")
            return
            
        self.modeling_code = modeling_code_result.get("code")
        logger.info("Modeling code generated successfully (not yet validated).")

        # Step 6: Run Assembler Agent to assemble, finalize, and run the code
        assembler_result = self.assembler_agent()
        if assembler_result.get("status") == "failed":
            logger.error(f"Final code assembly and execution failed: {assembler_result.get('error')}")
            return
        
        self.assembled_code = assembler_result.get("code")
        logger.info(f"Initial script generated and executed successfully.")

        logger.info("AutoML pipeline completed successfully!")

    def write_code_script(self, script, output_code_file):
        with open(output_code_file, "w") as file:
            file.write(script)

    def execute_code(self, code_to_execute: str, phase_name: str, attempt: int) -> dict:
        """
        Executes a string of Python code in a subprocess and saves the script,
        stdout, and stderr to a structured attempts folder.

        Args:
            code_to_execute: The Python code to run.
            phase_name: The name of the phase (e.g., "preprocessing", "assembler").
            attempt: The retry attempt number.

        Returns:
            A dictionary with execution status, stdout, and stderr.
        """
        # Create a structured directory for this attempt
        attempt_dir = Path(self.output_folder) / "attempts" / phase_name / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        # Define file paths for the script, stdout, and stderr
        script_path = attempt_dir / "code_generated.py"
        stdout_path = attempt_dir / "stdout.txt"
        stderr_path = attempt_dir / "stderr.txt"

        # Write the code to the script file
        self.write_code_script(code_to_execute, str(script_path))

        logger.info(f"Executing code from: {script_path}")

        try:
            # Execute the script using subprocess
            working_dir = str(Path(self.input_data_folder).parent)
            
            process = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                text=True,
                check=False,  # Do not raise exception on non-zero exit code
                cwd=working_dir,
                timeout=self.config.per_execution_timeout,
            )

            stdout = process.stdout
            stderr = process.stderr

            # Save stdout and stderr to their respective files
            with open(stdout_path, "w") as f:
                f.write(stdout)
            with open(stderr_path, "w") as f:
                f.write(stderr)

            if process.returncode == 0:
                logger.info("Code executed successfully.")
                return {"success": True, "stdout": stdout, "stderr": stderr}
            else:
                logger.error(f"Code execution failed with return code {process.returncode}.")
                full_error = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
                return {"success": False, "stdout": stdout, "stderr": full_error}

        except subprocess.TimeoutExpired as e:
            logger.error(f"Code execution timed out after {self.config.per_execution_timeout} seconds.")
            full_error = f"Timeout Error: Execution exceeded {self.config.per_execution_timeout} seconds.\n\nSTDOUT:\n{e.stdout or ''}\n\nSTDERR:\n{e.stderr or ''}"
            # Save partial output if available
            with open(stdout_path, "w") as f:
                f.write(e.stdout or "")
            with open(stderr_path, "w") as f:
                f.write(e.stderr or "")
            return {"success": False, "stdout": e.stdout, "stderr": full_error}
        except Exception as e:
            logger.error(f"An exception occurred during code execution: {e}")
            # Save exception to stderr file
            with open(stderr_path, "w") as f:
                f.write(str(e))
            return {"success": False, "stdout": "", "stderr": str(e)}


    def update_python_code(self):
        """Update the current Python code."""
        assert len(self.python_codes) == self.time_step
        assert len(self.python_file_paths) == self.time_step

        python_code = self.python_coder()

        python_file_path = os.path.join(self.iteration_folder, "generated_code.py")

        self.write_code_script(python_code, python_file_path)

        self.python_codes.append(python_code)
        self.python_file_paths.append(python_file_path)

    def update_bash_script(self):
        """Update the current bash script."""
        assert len(self.bash_scripts) == self.time_step

        bash_script = self.bash_coder()

        bash_file_path = os.path.join(self.iteration_folder, "execution_script.sh")

        self.write_code_script(bash_script, bash_file_path)

        self.bash_scripts.append(bash_script)

    def execute_code_old(self):
        planner_decision, planner_error_summary, planner_prompt, stderr, stdout = self.executer(
            code_to_execute=self.bash_script,
            code_to_analyze=self.python_code,
            task_description=self.task_description,
            data_prompt=self.data_prompt,
        )

        self.save_and_log_states(stderr, "stderr", add_uuid=False)
        self.save_and_log_states(stdout, "stdout", add_uuid=False)

        if planner_decision == "FIX":
            logger.brief(f"[bold red]Code generation failed in iteration[/bold red] {self.time_step}!")
            # Add suggestions to the error message to guide next iteration
            error_message = f"stderr: {stderr}\n\n" if stderr else ""
            error_message += (
                f"Error summary from planner (the error can appear in stdout if it's catched): {planner_error_summary}"
            )
            self.update_error_message(error_message=error_message)
            return False
        elif planner_decision == "FINISH":
            logger.brief(
                f"[bold green]Code generation successful after[/bold green] {self.time_step + 1} [bold green]iterations[/bold green]"
            )
            self.update_error_message(error_message="")
            return True
        else:
            logger.warning(f"###INVALID Planner Output: {planner_decision}###")
            self.update_error_message(error_message="")
            return False

    def update_error_message(self, error_message: str):
        """Update the current error message."""
        assert len(self.error_messages) == self.time_step
        self.error_messages.append(error_message)

    def save_and_log_states(self, content, save_name, add_uuid=False):
        if add_uuid:
            # Split filename and extension
            name, ext = os.path.splitext(save_name)
            # Generate 4-digit UUID (using first 4 characters of hex)
            uuid_suffix = str(uuid.uuid4()).replace("-", "")[:4]
            save_name = f"{name}_{uuid_suffix}{ext}"

        states_dir = os.path.join(self.output_folder, "states")
        os.makedirs(states_dir, exist_ok=True)
        output_file = os.path.join(states_dir, save_name)

        logger.info(f"Saving {output_file}...")
        with open(output_file, "w") as file:
            if content is not None:
                if isinstance(content, list):
                    # Join list elements with newlines
                    file.write("\n".join(str(item) for item in content))
                else:
                    # Handle as string (original behavior)
                    file.write(content)
            else:
                file.write("<None>")

    def log_agent_start(self, message: str):
        logger.brief(message)

    def log_agent_end(self, message: str):
        logger.brief(message)

    def report_token_usage(self):
        token_usage_path = os.path.join(self.output_folder, "token_usage.json")
        usage = ChatLLMFactory.get_total_token_usage(save_path=token_usage_path)
        total = usage["total"]
        logger.brief(
            f"Total tokens — input: {total['total_input_tokens']}, "
            f"output: {total['total_output_tokens']}, "
            f"sum: {total['total_tokens']}"
        )

        logger.info(f"Full token usage detail:\n{usage}")

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "retriever"):
            self.retriever.cleanup()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
