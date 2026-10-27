"""The command mcp: the commands served as tools of a local MCP server on stdio (oe_inferencex.mcp_server)."""


def cmd_mcp(args):
    """Serve the commands above as MCP tools on stdio, for an agent on the user's machine (oe_inferencex.mcp_server).
    The agent starts this; nobody types into it. Without the mcp extra it says how to install it."""
    from oe_inferencex import mcp_server
    try:
        server = mcp_server.build_server()
    except ImportError as exc:
        raise SystemExit(f"oe-inferencex mcp: {exc}") from None
    server.run("stdio")
    return 0


def add_mcp_parser(sub):
    m = sub.add_parser("mcp", help="serve these commands as tools to an agent on this machine (a local MCP server on "
                                    "stdio; needs the mcp extra)")
    m.set_defaults(func=cmd_mcp)
