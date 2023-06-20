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
import uuid
from colorama import Fore, Style
import jsonpickle
import os
from langchain.vectorstores import FAISS
from typing import NamedTuple

workspace_dir = Path('workspace')

class NodeData(NamedTuple):
    branch: str
    msg_history: List[BaseMessage]
    memory: FAISS

class NodeDataHistory(NamedTuple):
    data: NodeData
    commit_history: SortedDict[int, str]

class AutoGPT:
    """Agent class for interacting with Auto-GPT."""

    def __init__(
        self,
        ai_name: str,
        memory: VectorStoreRetriever,
        chain: LLMChain,
        output_parser: BaseAutoGPTOutputParser,
        tools: List[BaseTool],
        fais_hack: FAISS,
        feedback_tool: Optional[HumanInputRun] = None,
    ):
        self.ai_name = ai_name
        self.memory = memory
        self.full_message_history: List[BaseMessage] = []
        # full_message_history index to node_name
        self.commit_history: SortedDict[int, str] = {}
        # Graph key: "revert_count-loop_count"
        self.run_graph = nx.DiGraph()
        self.branch_uuid = uuid.uuid4().hex
        # node_name to NodeDataHistory
        self.run_history: dict(str, NodeDataHistory) = {}

        self.next_action_count = 0
        self.chain = chain
        self.output_parser = output_parser
        self.tools = tools
        self.feedback_tool = feedback_tool

        # make sure the path is the same as the one in BaseFileToolMixin
        self._root_dir = workspace_dir.resolve()
        self._root_dir.mkdir(exist_ok=True, parents=True)
        self._git = git.Repo.init(self._root_dir, bare=False)
        # Commit a master checkpoint in the repo is new.
        # Otherwise, just checkout master to have a clean start point
        if 'master' not in self._git.git.branch('--all').split():
            open(os.path.join(self._root_dir, 'test.txt'), 'w')
            self.add_git_checkpoint('master commit')
        else:
            self._git.git.reset('--hard')
            self._git.git.clean('-xdf')
            self._git.git.checkout('master')
        # hack
        self.faiss_hack = fais_hack
        self.faiss_hack_orig = copy.deepcopy(self.faiss_hack)

    @classmethod
    def from_llm_and_tools(
        cls,
        ai_name: str,
        ai_role: str,
        memory: VectorStoreRetriever,
        tools: List[BaseTool],
        llm: BaseChatModel,
        faiss_hack: FAISS,
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
            faiss_hack,
            feedback_tool=human_feedback_tool,
        )
    
    def load_from_node_data_history(self, node_data_history):
        node_data: NodeData = node_data_history.data
        branches = self._git.git.branch("--all").split()
        if node_data.branch not in branches:
            # Should the be a warning instead of an error?
            # File state may not match if we treat it as a warning
            raise "Git branch associated with data does not exist"

        self.commit_history = copy.deepcopy(node_data_history.commit_history)
        self.full_message_history = copy.deepcopy(node_data.msg_history)
        self.faiss_hack = copy.deepcopy(node_data.memory)

        self._git.git.checkout(node_data.branch)
    
    # This function will delete any uncommitted changes!
    def reset_to_checkpoint(self, index):
        self._git.git.reset('--hard')
        self._git.git.clean('-xdf')
        node_name = None
        if index <= 0:
            # key 0 should always exist.... TODO make this better
            node_name = self.commit_history.get(0)
        else:
            idx, node_name = self.commit_history.popitem()
            while idx > index:
                idx, node_name = self.commit_history.popitem()

        if node_name is None:
            raise "No Node name found at index"
        self.load_from_node_data_history(self.run_history[node_name])

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
    
    # returns (revert_count, loop_count)
    def rehydrate_from_files(self):
        dot_file = os.path.join(self._root_dir, 'graph.dot')
        data_dir = os.path.join(self._root_dir, 'data')
        if not os.path.isfile(dot_file) or not os.path.exists(data_dir):
            raise "Rehydrate files do not exist"
        self.run_graph = nx.DiGraph(nx.nx_agraph.read_dot(dot_file))
        nodes = list(self.run_graph.nodes())
        if not nodes or len(nodes) == 0:
            return (0,0)
        nodes.sort()
        final_node = nodes[-1]
        if final_node == "":
            raise "Invalid run graph"
        if final_node == 'end':
            raise "Cannot load a completed run"
        
        for node in nodes:
            with open(os.path.join(data_dir, node), "r") as f:
                # the docstore needs to be regenerated from scratch
                node_history: NodeDataHistory = jsonpickle.decode(f.read(), keys=True)
                self.run_history[node] = NodeDataHistory(
                    NodeData(node_history.data.branch, 
                             node_history.data.msg_history, 
                             copy.deepcopy(self.faiss_hack_orig)), 
                    node_history.commit_history)
                cur_node_data: NodeDataHistory = self.run_history[node]
                
                for doc in node_history.data.memory.docstore._dict.values():
                    retriever = cur_node_data.data.memory.as_retriever()
                    retriever.add_documents([doc])

        self.load_from_node_data_history(self.run_history[final_node])

        node_iteration = final_node.split("-")
        if len(node_iteration) != 2:
            raise "Invalid run graph"
        if not node_iteration[0].isnumeric() or not node_iteration[1].isnumeric():
            raise "Invalid run graph"
        return (int(node_iteration[0]), int(node_iteration[1]))
    
    def rehydrate_from_branch(self, branch):
        branches = self._git.git.branch("--all").split()
        if branch not in branches:
            # Should the be a warning instead of an error?
            # File state may not match if we treat it as a warning
            raise "Git branch associated with data does not exist"
        self._git.git.checkout(branch)
        return self.rehydrate_from_files()

    def write_data_files(self):
        dot_file = os.path.join(self._root_dir, 'graph.dot')
        data_dir = os.path.join(self._root_dir, 'data')
        Path(data_dir).mkdir(exist_ok=True, parents=True)
        nx.drawing.nx_agraph.write_dot(self.run_graph, dot_file)
        for node, data in self.run_history.items():
            data_file = os.path.join(data_dir, node)
            with open(data_file, "w") as f:
                f.write(jsonpickle.encode(data, keys=True))

    # prev_node = None for root node
    def add_node(self, new_node, prev_node, branch_name):
        self.commit_history[len(self.full_message_history)] = new_node

        # commit history can't be attached directly to the node since they get too long.
        # need to save it off in another file
        self.run_graph.add_node(new_node)
        if prev_node is not None:
            self.run_graph.add_edge(prev_node, new_node)
        data = NodeData(branch_name, self.full_message_history, self.faiss_hack)
        self.run_history[new_node] = copy.deepcopy(NodeDataHistory(data, self.commit_history))


    def run(self,
            goals: List[str],
            rehydrate_branch: Optional(str) = None
            ) -> str:
        user_input = (
            "Determine which next command to use, "
            "and respond using the format specified above:"
        )
        # Interaction Loop
        loop_count = 0
        revert_count = 0
        continuous = 0
        goals_hash = hex(hash(str(goals)))[2:]

        branch_name = "{0}_{1}_{2}".format(self.ai_name, goals_hash, self.branch_uuid)
        prev_node_name = "{0}-{1}".format(revert_count, loop_count)

        if(rehydrate_branch is not None):
            revert_count, loop_count = self.rehydrate_from_branch(rehydrate_branch)
            prev_node_name = "{0}-{1}".format(revert_count, loop_count)
        else:
            # create root node
            self.add_git_checkpoint("clean commit", branch_name)
            self.add_node(prev_node_name, None, branch_name)

        cur_branch_name = "{0}_{1}".format(branch_name, prev_node_name)        
        while True:
            # Discontinue if continuous limit is reached
            loop_count += 1
            current_node_name = "{0}-{1}".format(revert_count, loop_count)
            cur_branch_name = "{0}_{1}".format(branch_name, current_node_name)
            self.write_data_files()
            self.add_git_checkpoint("commit", cur_branch_name)
            self.add_node(current_node_name, prev_node_name, cur_branch_name)
            prev_node_name = current_node_name

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
                # Write out the last node
                cur_branch_name = "{0}_end".format(branch_name)
                self.add_node('end', prev_node_name, cur_branch_name)
                self.write_data_files()
                self.add_git_checkpoint("commit", cur_branch_name)

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

            if self.feedback_tool is not None and continuous == 0:
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
                        print(Fore.GREEN + "{}: ".format(i) + Style.RESET_ALL + message)
                    feedback = f"{self.feedback_tool.run('Index: ')}"
                    feedback = int(feedback)
                    loop_count = feedback
                    revert_count += 1
                    counter = 1
                    prev_node_name = "{0}-{1}".format(revert_count-counter, loop_count)
                    while not self.run_graph.has_node(prev_node_name):
                        counter += 1
                        prev_node_name = "{0}-{1}".format(revert_count-counter, loop_count)
                        if counter > revert_count:
                            # This shouldn't ever happen..
                            raise
                    
                    idx = keys[feedback]
                    self.reset_to_checkpoint(idx)
                    continue
                elif len(feedback) > 0 and feedback.split()[0] == 'c':
                    continuous = [int(i) for i in feedback.split() if i.isdigit()][0] + 1
                    feedback = ''
                memory_to_add += '\n'+feedback

            self.memory.add_documents([Document(page_content=memory_to_add)])
            self.full_message_history.append(SystemMessage(content=result))
            if continuous > 0:
                continuous -= 1
