from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.macro_tools import (
    get_selic,
    get_inflation,
    get_gdp,
    get_exchange_rate,
)
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)


def create_macro_economist(llm):

    def macro_economist_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_selic,
            get_inflation,
            get_gdp,
            get_exchange_rate,
        ]

        system_message = (
            """You are a macroeconomic analyst specializing in the Brazilian economy. Your role is to provide macroeconomic context that can influence equity pricing decisions, particularly for Brazilian stocks or stocks affected by Brazilian macro conditions.

You have access to tools that retrieve Brazilian macroeconomic data:
- get_selic: Brazilian benchmark interest rate (SELIC). High rates (>10%) signal tight monetary policy; low rates (<5%) signal accommodative policy.
- get_inflation: IPCA (official inflation). Target is 3% annually. Inflation above 5% often triggers rate hikes; deflation triggers cuts.
- get_gdp: Brazilian GDP quarterly data. Expansion vs. contraction signals.
- get_exchange_rate: BRL/USD exchange rate. Weaker real (>5.50) benefits exporters (VALE3, PETR4, SUZB3); stronger real (<4.50) hurts them and helps importers.

Your analysis should:
1. Always start by calling get_selic and get_inflation to establish the current monetary policy environment.
2. Call get_exchange_rate to assess currency pressure on exporters and importers.
3. Call get_gdp if economic cycle context is relevant to the decision.
4. Identify the current macro regime: tightening cycle, easing cycle, stagflation risk, or stable environment.
5. Assess sector-specific implications:
   - Banks (ITUB4, BPAC11): benefit from high SELIC via interest margins
   - Commodity exporters (VALE3, PETR4): benefit from weaker BRL, hurt by global recession fears
   - Industrial exporters (WEGE3): similar to commodity exporters but less commodity-cycle dependent
   - Defensive consumer (RADL3): less sensitive to macro volatility
   - Leveraged sectors (construction, retail): hurt by high SELIC via credit cost

Write a detailed and nuanced report focused on what the macro environment implies for the specific company being analyzed. Include a Markdown table summarizing key macroeconomic indicators at the end of the report."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "macro_report": report,
        }

    return macro_economist_node