import dataclasses


@dataclasses.dataclass(frozen=True)
class KmeansMetadata:
    shape: (int, int)
    dtype: type


class Tags:
    KMEANS_METADATA = 1
    KMEANS_DATA = 2
