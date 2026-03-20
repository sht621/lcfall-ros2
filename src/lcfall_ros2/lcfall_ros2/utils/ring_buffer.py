"""固定長リングバッファ.

inference_node で 48 フレーム分の PreprocessedFrame を蓄積するために使用する。
"""

from __future__ import annotations

from typing import Generic, List, Optional, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """固定長リングバッファ.

    容量を超えて追加すると、最も古い要素が上書きされる。
    stride カウントによる推論トリガー判定もサポートする。
    """

    def __init__(self, capacity: int) -> None:
        """リングバッファを初期化.

        Args:
            capacity: バッファの最大長 (例: 48)。
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")

        self._capacity: int = capacity
        self._buffer: List[Optional[T]] = [None] * capacity
        self._write_pos: int = 0
        self._count: int = 0
        self._total_frames: int = 0

    # ------------------------------------------------------------------
    # 公開プロパティ
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """バッファ容量."""
        return self._capacity

    @property
    def count(self) -> int:
        """現在蓄積されているフレーム数 (capacity 以下)."""
        return self._count

    @property
    def total_frames(self) -> int:
        """初期化以降の総受信フレーム数."""
        return self._total_frames

    @property
    def is_full(self) -> bool:
        """バッファが満杯かどうか."""
        return self._count >= self._capacity

    # ------------------------------------------------------------------
    # 公開メソッド
    # ------------------------------------------------------------------

    def append(self, item: T) -> None:
        """要素を追加.

        バッファが満杯の場合、最も古い要素が上書きされる。

        Args:
            item: 追加する要素。
        """
        self._buffer[self._write_pos] = item
        self._write_pos = (self._write_pos + 1) % self._capacity
        self._count = min(self._count + 1, self._capacity)
        self._total_frames += 1

    def get_ordered(self) -> List[T]:
        """時系列順に並んだリストを返す.

        最も古い要素が先頭、最も新しい要素が末尾。

        Returns:
            蓄積済み要素のリスト。バッファが満杯でなくても、
            蓄積済み分だけ返す。
        """
        if self._count < self._capacity:
            # まだ満杯でない → 挿入順に先頭から
            return [
                self._buffer[i]
                for i in range(self._count)
                if self._buffer[i] is not None
            ]

        # 満杯 → write_pos が最も古い要素を指す
        result: List[T] = []
        for i in range(self._capacity):
            idx = (self._write_pos + i) % self._capacity
            item = self._buffer[idx]
            if item is not None:
                result.append(item)
        return result

    def should_infer(self, stride: int) -> bool:
        """推論を実行すべきかどうかを判定.

        バッファが満杯かつ、total_frames が stride の倍数のときに True。

        Args:
            stride: 推論間隔 (フレーム数)。

        Returns:
            True なら推論を実行すべき。
        """
        if not self.is_full:
            return False
        return (self._total_frames % stride) == 0

    def clear(self) -> None:
        """バッファをクリア."""
        self._buffer = [None] * self._capacity
        self._write_pos = 0
        self._count = 0
        self._total_frames = 0
