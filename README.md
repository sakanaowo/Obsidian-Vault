# What are you doing in here? 🤨

Build pass nghĩa là Bước 6.1 — Design Tokens hoàn thành. Theo thứ tự implementation đã chốt, bây giờ làm Bước 6.2 — Basic Calculator UI: dựng màn hình hoàn chỉnh nhưng chưa viết logic tính toán.  

Bước 6.2 — Basic Calculator UI

Mục tiêu sau bước này:

App mở lên
↓
Background đen
↓
History             Mode
↓
Display "0"
↓
⌫    AC    %    ÷
7     8    9    ×
4     5    6    −
1     2    3    +
±     0    .    =

Các nút chưa cần hoạt động.

⸻

6.2.1 — Tạo icon History

Trong:

app
└── src
    └── main
        └── res
            └── drawable

Tạo:

ic_history.xml

Nội dung:

<?xml version="1.0" encoding="utf-8"?>
<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="24dp"
    android:height="24dp"
    android:viewportWidth="24"
    android:viewportHeight="24">
    <path
        android:fillColor="@color/calculator_text_primary"
        android:pathData="M13,3C8.03,3 4,7.03 4,12H1L5,16L9,12H6C6,8.13 9.13,5 13,5C16.87,5 20,8.13 20,12C20,15.87 16.87,19 13,19C9.74,19 6.99,16.77 6.21,13.75L4.27,14.25C5.27,18.13 8.79,21 13,21C17.97,21 22,16.97 22,12C22,7.03 17.97,3 13,3M12,7V13L17,16L18,14.5L14,12.1V7H12Z" />
</vector>

⸻

6.2.2 — Tạo icon Mode

Tạo:

ic_calculator_mode.xml

Nội dung:

<?xml version="1.0" encoding="utf-8"?>
<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="24dp"
    android:height="24dp"
    android:viewportWidth="24"
    android:viewportHeight="24">
    <path
        android:fillColor="@color/calculator_text_primary"
        android:pathData="M7,2H17C18.1,2 19,2.9 19,4V20C19,21.1 18.1,22 17,22H7C5.9,22 5,21.1 5,20V4C5,2.9 5.9,2 7,2M7,4V8H17V4H7M8,10V12H10V10H8M11,10V12H13V10H11M14,10V12H16V10H14M8,13V15H10V13H8M11,13V15H13V13H11M14,13V18H16V13H14M8,16V18H10V16H8M11,16V18H13V16H11Z" />
</vector>

Sau đó drawable có thêm:

drawable/
├── bg_button_number.xml
├── bg_button_function.xml
├── bg_button_operator.xml
├── bg_top_button.xml
├── ic_history.xml
└── ic_calculator_mode.xml

⸻

6.2.3 — Thay activity_main.xml

Mở:

app/src/main/res/layout/activity_main.xml

Thay toàn bộ bằng:

<?xml version="1.0" encoding="utf-8"?>
<FrameLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:id="@+id/main"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:background="@color/calculator_background">
    <include
        android:id="@+id/basicCalculatorLayout"
        layout="@layout/layout_basic_calculator"
        android:layout_width="match_parent"
        android:layout_height="match_parent" />
</FrameLayout>

Quan trọng

Giữ:

android:id="@+id/main"

Vì MainActivity.java của template Empty Views Activity có thể đang dùng:

findViewById(R.id.main)

để xử lý system bar inset.

Nếu bỏ ID này, app có thể crash khi mở.

⸻

6.2.4 — Thay layout_basic_calculator.xml

Mở:

app/src/main/res/layout/layout_basic_calculator.xml

Thay toàn bộ bằng:

<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:id="@+id/basicCalculatorRoot"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:background="@color/calculator_background"
    android:paddingStart="@dimen/screen_padding_horizontal"
    android:paddingEnd="@dimen/screen_padding_horizontal"
    android:paddingTop="@dimen/screen_padding_vertical"
    android:paddingBottom="@dimen/screen_padding_vertical">
    <!-- ====================================================== -->
    <!-- TOP BAR                                                -->
    <!-- ====================================================== -->
    <LinearLayout
        android:id="@+id/topBar"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:gravity="center_vertical"
        android:orientation="horizontal">
        <ImageButton
            android:id="@+id/btnHistory"
            android:layout_width="@dimen/top_button_size"
            android:layout_height="@dimen/top_button_size"
            android:background="@drawable/bg_top_button"
            android:contentDescription="@string/history"
            android:padding="14dp"
            android:scaleType="centerInside"
            android:src="@drawable/ic_history" />
        <!-- Flexible space -->
        <Space
            android:layout_width="0dp"
            android:layout_height="1dp"
            android:layout_weight="1" />
        <ImageButton
            android:id="@+id/btnMode"
            android:layout_width="@dimen/top_button_size"
            android:layout_height="@dimen/top_button_size"
            android:background="@drawable/bg_top_button"
            android:contentDescription="@string/calculator_mode"
            android:padding="14dp"
            android:scaleType="centerInside"
            android:src="@drawable/ic_calculator_mode" />
    </LinearLayout>
    <!-- ====================================================== -->
    <!-- DISPLAY AREA                                           -->
    <!-- Chiếm toàn bộ khoảng trống giữa Top Bar và Keypad      -->
    <!-- ====================================================== -->
    <LinearLayout
        android:id="@+id/displayArea"
        android:layout_width="match_parent"
        android:layout_height="0dp"
        android:layout_weight="1"
        android:gravity="bottom|end"
        android:orientation="vertical"
        android:paddingStart="@dimen/spacing_8"
        android:paddingEnd="@dimen/spacing_8"
        android:paddingBottom="@dimen/spacing_16">
        <TextView
            android:id="@+id/tvExpression"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:ellipsize="start"
            android:gravity="end"
            android:maxLines="1"
            android:text=""
            android:textColor="@color/calculator_text_secondary"
            android:textSize="@dimen/text_expression" />
        <TextView
            android:id="@+id/tvResult"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:ellipsize="start"
            android:gravity="end"
            android:maxLines="1"
            android:text="0"
            android:textColor="@color/calculator_text_primary"
            android:textSize="@dimen/text_result" />
    </LinearLayout>
    <!-- ====================================================== -->
    <!-- KEYPAD                                                 -->
    <!-- ====================================================== -->
    <LinearLayout
        android:id="@+id/keypad"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:orientation="vertical">
        <!-- ================= ROW 1 ================= -->
        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="horizontal">
            <Button
                android:id="@+id/btnBackspace"
                style="@style/CalculatorButtonFunction"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_margin="4dp"
                android:text="⌫" />
            <Button
                android:id="@+id/btnClear"
                style="@style/CalculatorButtonFunction"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_margin="4dp"
                android:text="AC" />
            <Button
                android:id="@+id/btnPercent"
                style="@style/CalculatorButtonFunction"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_margin="4dp"
                android:text="%" />
            <Button
                android:id="@+id/btnDivide"
                style="@style/CalculatorButtonOperator"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_margin="4dp"
                android:text="÷" />
        </LinearLayout>
        <!-- ================= ROW 2 ================= -->
        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="horizontal">
            <Button
                android:id="@+id/btn7"
                style="@style/CalculatorButtonNumber"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_margin="4dp"
                android:text="7" />
            <Button
                android:id="@+id/btn8"
                style="@style/CalculatorButtonNumber"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_margin="4dp"
                android:text="8" />
            <Button
                android:id="@+id/btn9"
                style="@style/CalculatorButtonNumber"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_margin="4dp"
                android:text="9" />
            <Button
                android:id="@+id/btnMultiply"
                style="@style/CalculatorButtonOperator"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_margin="4dp"
                android:text="×" />
        </LinearLayout>
        <!-- ================= ROW 3 ================= -->
        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="horizontal">
            <Button
                android:id="@+id/btn4"
                style="@style/CalculatorButtonNumber"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_margin="4dp"
                android:text="4" />
            <Button
                android:id="@+id/btn5"
                style="@style/CalculatorButtonNumber"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_margin="4dp"
                android:text="5" />
            <Button
                android:id="@+id/btn6"
                style="@style/CalculatorButtonNumber"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_margin="4dp"
                android:text="6" />
            <Button
                android:id="@+id/btnSubtract"
                style="@style/CalculatorButtonOperator"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_margin="4dp"
                android:text="−" />
        </LinearLayout>
        <!-- ================= ROW 4 ================= -->
        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="horizontal">
            <Button
                android:id="@+id/btn1"
                style="@style/CalculatorButtonNumber"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_margin="4dp"
                android:text="1" />
            <Button
                android:id="@+id/btn2"
                style="@style/CalculatorButtonNumber"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_margin="4dp"
                android:text="2" />
            <Button
                android:id="@+id/btn3"
                style="@style/CalculatorButtonNumber"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_margin="4dp"
                android:text="3" />
            <Button
                android:id="@+id/btnAdd"
                style="@style/CalculatorButtonOperator"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_margin="4dp"
                android:text="+" />
        </LinearLayout>
        <!-- ================= ROW 5 ================= -->
        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="horizontal">
            <Button
                android:id="@+id/btnToggleSign"
                style="@style/CalculatorButtonFunction"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_margin="4dp"
                android:text="±" />
            <Button
                android:id="@+id/btn0"
                style="@style/CalculatorButtonNumber"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_margin="4dp"
                android:text="0" />
            <Button
                android:id="@+id/btnDecimal"
                style="@style/CalculatorButtonNumber"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_margin="4dp"
                android:text="." />
            <Button
                android:id="@+id/btnEquals"
                style="@style/CalculatorButtonOperator"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_margin="4dp"
                android:text="=" />
        </LinearLayout>
    </LinearLayout>
</LinearLayout>

⸻

6.2.5 — Kiểm tra MainActivity.java

Ở bước này không cần sửa logic.

Chỉ kiểm tra nó vẫn có:

setContentView(R.layout.activity_main);

Ví dụ template có thể đang là:

package com.example.calculator;
import android.os.Bundle;
import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(
                findViewById(R.id.main),
                (v, insets) -> {
                    Insets systemBars =
                            insets.getInsets(
                                    WindowInsetsCompat.Type.systemBars()
                            );
                    v.setPadding(
                            systemBars.left,
                            systemBars.top,
                            systemBars.right,
                            systemBars.bottom
                    );
                    return insets;
                }
        );
    }
}

Nếu project của bạn có code mặc định gần giống vậy thì giữ nguyên.

Không copy đoạn trên nếu file hiện tại đã build tốt.

⸻

6.2.6 — Build

Chạy:

Build
→ Make Project

Điều kiện:

✓ XML compile
✓ Vector compile
✓ Không missing resource
✓ MainActivity compile

⸻

6.2.7 — Run app

Bây giờ Run:

▶ Run 'app'

Kết quả mong đợi:

┌────────────────────────────────┐
│ ◷                          ▦   │
│                                │
│                                │
│                                │
│                             0  │
│                                │
│  ⌫     AC     %       ÷        │
│                                │
│  7      8     9       ×        │
│                                │
│  4      5     6       −        │
│                                │
│  1      2     3       +        │
│                                │
│  ±      0     .       =        │
└────────────────────────────────┘

Visual checklist:

[ ] Background đen
[ ] History button tròn bên trái
[ ] Mode button tròn bên phải
[ ] Display "0" căn phải
[ ] Có đúng 5 hàng × 4 nút
[ ] Number buttons xám đậm
[ ] Function buttons xám sáng
[ ] Operators màu cam
[ ] Các nút bo tròn
[ ] Các cột có chiều rộng bằng nhau
[ ] Không có button bị cắt
[ ] Không overflow khỏi màn hình
[ ] Status/navigation bar không che UI

⸻

6.2.8 — Các nút chưa hoạt động là đúng

Ở thời điểm này:

Tap 7
→ không thay đổi
Tap +
→ không thay đổi
Tap =
→ không thay đổi
Tap AC
→ không thay đổi

Đây không phải lỗi.

Ta đang tách implementation thành:

UI
↓
Engine
↓
Connection

Không viết click logic trực tiếp vào UI trước khi engine tồn tại.

⸻

Nếu button bị màu tím hoặc màu theme thay vì màu đã định nghĩa

Một số Material theme có thể tint <Button>.

Trong styles.xml, đảm bảo CalculatorButtonBase có:

<item name="android:backgroundTint">@null</item>

Nếu vẫn bị tint, thêm trực tiếp vào các button:

android:backgroundTint="@null"

Nhưng chỉ làm khi thực tế gặp lỗi màu.

⸻

Nếu chữ nút bị cắt

Ví dụ:

AC
±
⌫

bị lệch hoặc cắt, bổ sung trong CalculatorButtonBase:

<item name="android:minWidth">0dp</item>
<item name="android:minHeight">0dp</item>
<item name="android:includeFontPadding">false</item>

Thành:

<style name="CalculatorButtonBase">
    <item name="android:layout_height">
        @dimen/calculator_button_height
    </item>
    <item name="android:textColor">
        @color/calculator_text_primary
    </item>
    <item name="android:fontFamily">
        sans
    </item>
    <item name="android:textAllCaps">
        false
    </item>
    <item name="android:gravity">
        center
    </item>
    <item name="android:padding">
        0dp
    </item>
    <item name="android:minWidth">
        0dp
    </item>
    <item name="android:minHeight">
        0dp
    </item>
    <item name="android:includeFontPadding">
        false
    </item>
    <item name="android:backgroundTint">
        @null
    </item>
</style>

⸻

Checkpoint sau Bước 6.2

Khi Build + Run pass:

STEP 6 — IMPLEMENTATION
[✓] 6.1 Design Tokens
[✓] 6.2 Basic Calculator UI
     ├── Top Bar
     ├── Display
     ├── 5 × 4 Keypad
     ├── Number style
     ├── Function style
     └── Operator style
[ ] 6.3 Calculator State
        ↓
[ ] 6.4 Basic Calculator Engine
        ↓
[ ] 6.5 Connect MainActivity ↔ Engine
        ↓
[ ] 6.6 Basic Functional Test

Sau khi màn hình này Build + Run đúng, bước kế tiếp sẽ là 6.3 — triển khai CalculatorState.java và CalculatorEngine.java trước, hoàn toàn độc lập với Android UI. Sau đó mới nối các btn0...btn9, operator, AC, %, ±, backspace và = vào MainActivity.</resources>