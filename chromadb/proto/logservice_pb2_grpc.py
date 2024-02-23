# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from chromadb.proto import logservice_pb2 as chromadb_dot_proto_dot_logservice__pb2


class LogServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.PushLogs = channel.unary_unary(
            "/chroma.LogService/PushLogs",
            request_serializer=chromadb_dot_proto_dot_logservice__pb2.PushLogsRequest.SerializeToString,
            response_deserializer=chromadb_dot_proto_dot_logservice__pb2.PushLogsResponse.FromString,
        )
        self.PullLogs = channel.unary_unary(
            "/chroma.LogService/PullLogs",
            request_serializer=chromadb_dot_proto_dot_logservice__pb2.PullLogsRequest.SerializeToString,
            response_deserializer=chromadb_dot_proto_dot_logservice__pb2.PullLogsResponse.FromString,
        )


class LogServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def PushLogs(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def PullLogs(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_LogServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "PushLogs": grpc.unary_unary_rpc_method_handler(
            servicer.PushLogs,
            request_deserializer=chromadb_dot_proto_dot_logservice__pb2.PushLogsRequest.FromString,
            response_serializer=chromadb_dot_proto_dot_logservice__pb2.PushLogsResponse.SerializeToString,
        ),
        "PullLogs": grpc.unary_unary_rpc_method_handler(
            servicer.PullLogs,
            request_deserializer=chromadb_dot_proto_dot_logservice__pb2.PullLogsRequest.FromString,
            response_serializer=chromadb_dot_proto_dot_logservice__pb2.PullLogsResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "chroma.LogService", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class LogService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def PushLogs(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/chroma.LogService/PushLogs",
            chromadb_dot_proto_dot_logservice__pb2.PushLogsRequest.SerializeToString,
            chromadb_dot_proto_dot_logservice__pb2.PushLogsResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def PullLogs(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/chroma.LogService/PullLogs",
            chromadb_dot_proto_dot_logservice__pb2.PullLogsRequest.SerializeToString,
            chromadb_dot_proto_dot_logservice__pb2.PullLogsResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
