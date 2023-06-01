from __future__ import annotations

from typing import List, Optional

from pydantic import ValidationError
import git
from pathlib import Path
from sortedcontainers import SortedDict

from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.experimental.autonomous_agents.autogpt.output_parser import (
    AutoGPTOutputParser,
    BaseAutoGPTOutputParser,
)
from langchain.experimental.autonomous_agents.autogpt.prompt import AutoGPTPrompt
from langchain.experimental.autonomous_agents.autogpt.prompt_generator import (
    FINISH_NAME,
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    Document,
    HumanMessage,
    SystemMessage,
)
from langchain.tools.base import BaseTool
from langchain.tools.human.tool import HumanInputRun
from langchain.vectorstores.base import VectorStoreRetriever

import copy
import networkx as nx
import matplotlib.pyplot as plt

class AutoGPT:
    """Agent class for interacting with Auto-GPT."""

    def __init__(
        self,
        ai_name: str,
        memory: VectorStoreRetriever,
        chain: LLMChain,
        output_parser: BaseAutoGPTOutputParser,
        tools: List[BaseTool],
        feedback_tool: Optional[HumanInputRun] = None,
    ):
        self.ai_name = ai_name
        self.memory = memory
        self.full_message_history: List[BaseMessage] = []
        self.commit_history: SortedDict[int, (git.Commit, List[BaseMessage], VectorStoreRetriever)] = {}
        # Graph key: "revert_count-loop_count"
        self.run_history = nx.DiGraph()

        self.next_action_count = 0
        self.chain = chain
        self.output_parser = output_parser
        self.tools = tools
        self.feedback_tool = feedback_tool

        # make sure the path is the same as the one in BaseFileToolMixin
        self._root_dir = Path("workspace").resolve()
        self._root_dir.mkdir(exist_ok=True, parents=True)
        with open('workspace/test.txt', 'w') as f:
            pass
        self._git = git.Repo.init(self._root_dir, bare=False)
        self.add_git_checkpoint('master commit')

    @classmethod
    def from_llm_and_tools(
        cls,
        ai_name: str,
        ai_role: str,
        memory: VectorStoreRetriever,
        tools: List[BaseTool],
        llm: BaseChatModel,
        human_in_the_loop: bool = False,
        output_parser: Optional[BaseAutoGPTOutputParser] = None,
    ) -> AutoGPT:
        prompt = AutoGPTPrompt(
            ai_name=ai_name,
            ai_role=ai_role,
            tools=tools,
            input_variables=["memory", "messages", "goals", "user_input"],
            token_counter=llm.get_num_tokens,
        )
        human_feedback_tool = HumanInputRun() if human_in_the_loop else None
        chain = LLMChain(llm=llm, prompt=prompt)

        return cls(
            ai_name,
            memory,
            chain,
            output_parser or AutoGPTOutputParser(),
            tools,
            feedback_tool=human_feedback_tool,
        )
    
    def reset_to_checkpoint(self, index):
        if index <= 0:
            # key 0 should always exist.... TODO make this better
            (cid, msg_history, memory) = self.commit_history.get(0)
            self._git.git.checkout(cid)
            self.full_message_history = copy.deepcopy(msg_history)
            self.memory = copy.deepcopy(memory)
            self.commit_history.clear()
            return

        # Remove items up to and including the checkpointed index

        idx, (cid, msg_history, memory) = self.commit_history.popitem()
        while idx > index:
            idx, (cid, msg_history, memory) = self.commit_history.popitem()

        self.full_message_history = copy.deepcopy(msg_history)
        self.memory = copy.deepcopy(memory)
        self._git.git.checkout(cid)

    def add_git_checkpoint(self, commit_message, branch=None):
            cid = None
            if commit_message is None:
                commit_message = 'checkpoint'
            if branch is not None and branch.strip() != '':
                branches = self._git.git.branch("--all").split()
                if branch not in branches:
                    self._git.git.branch(branch)
                self._git.git.checkout(branch)
            if self._git.is_dirty(untracked_files=True):
                self._git.git.add('--all')
                cid = self._git.git.commit('-am', commit_message)
            else:
                cid = self._git.head.commit
            return cid


    def run(self, goals: List[str]) -> str:
        user_input = (
            "Determine which next command to use, "
            "and respond using the format specified above:"
        )
        # Interaction Loop
        loop_count = 0
        revert_count = 0
        # add initial checkpoint. Switch to this loop's branch
        self.add_git_checkpoint('initial checkpoint', self.ai_name)
        # create root node
        prev_node_name = "{0}-{1}".format(revert_count, loop_count)
        self.run_history.add_node(prev_node_name)
        while True:
            # Discontinue if continuous limit is reached
            loop_count += 1
            # assume we're already on the correct branch?
            current_node_name = "{0}-{1}".format(revert_count, loop_count)
            cid = self.add_git_checkpoint("start " + current_node_name)
            self.commit_history[len(self.full_message_history)] = (cid, copy.deepcopy(self.full_message_history), copy.deepcopy(self.memory))

            self.run_history.add_node(current_node_name, data=copy.deepcopy(self.commit_history))
            self.run_history.add_edge(prev_node_name, current_node_name)
            prev_node_name = current_node_name
            #nx.drawing.nx_agraph.write_dot(self.run_history, "graph.dot")

            # will chain.run try to modify the message history? Lets hope not
            # Send message to AI, get response
            assistant_reply = self.chain.run(
                goals=goals,
                messages=self.full_message_history,
                memory=self.memory,
                user_input=user_input,
            )

            # Print Assistant thoughts
            print(assistant_reply)
            self.full_message_history.append(HumanMessage(content=user_input))
            self.full_message_history.append(AIMessage(content=assistant_reply))

            # Get command name and arguments
            action = self.output_parser.parse(assistant_reply)
            tools = {t.name: t for t in self.tools}
            if action.name == FINISH_NAME:
                return action.args["response"]
            if action.name in tools:
                tool = tools[action.name]
                try:
                    observation = tool.run(action.args)
                except ValidationError as e:
                    observation = (
                        f"Validation Error in args: {str(e)}, args: {action.args}"
                    )
                except Exception as e:
                    observation = (
                        f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"
                    )
                result = f"Command {tool.name} returned: {observation}"
            elif action.name == "ERROR":
                result = f"Error: {action.args}. "
            else:
                result = (
                    f"Unknown command '{action.name}'. "
                    f"Please refer to the 'COMMANDS' list for available "
                    f"commands and only respond in the specified JSON format."
                )

            memory_to_add = (
                f"Assistant Reply: {assistant_reply} " f"\nResult: {result} "
            )

            # assume we're already on the correct branch?
            # need to do this in case a file was already written..
            self.add_git_checkpoint('mid loop ' + str(loop_count))
            #self.commit_history[len(self.full_message_history)] = cid

            if self.feedback_tool is not None:
                feedback = f"{self.feedback_tool.run('Input: ')}"
                if feedback in {"q", "stop"}:
                    print("EXITING")
                    return "EXITING"
                elif feedback in {"revert", "r", "reset"}:
                    # The only valid revert points are the ones at the beginning
                    # of each loop. We will actually revert to 1 step before the
                    # actual checkpoint, which will get re-added at the beginning
                    # of the next loop iteration
                    keys = list(self.commit_history.keys())
                    for i in range(len(keys)):
                        message = '**full reset**'
                        if i > 0:
                            message = self.full_message_history[keys[i] - 1].content
                        print("{0}: {1}".format(i, message))
                    feedback = f"{self.feedback_tool.run('Index: ')}"
                    feedback = int(feedback)
                    loop_count = feedback
                    revert_count += 1
                    counter = 1
                    prev_node_name = "{0}-{1}".format(revert_count-counter, loop_count)
                    while not self.run_history.has_node(prev_node_name):
                        counter += 1
                        prev_node_name = "{0}-{1}".format(revert_count-counter, loop_count)
                        if counter > revert_count:
                            # This shouldn't ever happen..
                            raise
                    
                    idx = keys[feedback]
                    self.reset_to_checkpoint(idx)
                    continue
                memory_to_add += '\n'+feedback

            self.memory.add_documents([Document(page_content=memory_to_add)])
            self.full_message_history.append(SystemMessage(content=result))
