# What are you doing in here? 🤨

## File: `app/src/main/res/values/styles.xml`

```xml
<?xml version="1.0" encoding="utf-8"?>
<resources>

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

        <!-- Nếu template/theme báo lỗi dòng này thì có thể bỏ -->
        <item name="android:backgroundTint">
            @null
        </item>

    </style>


    <style
        name="CalculatorButtonNumber"
        parent="CalculatorButtonBase">

        <item name="android:background">
            @drawable/bg_button_number
        </item>

        <item name="android:textSize">
            @dimen/text_button_number
        </item>

    </style>


    <style
        name="CalculatorButtonFunction"
        parent="CalculatorButtonBase">

        <item name="android:background">
            @drawable/bg_button_function
        </item>

        <item name="android:textSize">
            @dimen/text_button_function
        </item>

    </style>


    <style
        name="CalculatorButtonOperator"
        parent="CalculatorButtonBase">

        <item name="android:background">
            @drawable/bg_button_operator
        </item>

        <item name="android:textSize">
            @dimen/text_button_operator
        </item>

    </style>

</resources>
```

---

## File: `app/src/main/res/values/strings.xml`

```xml
<resources>

    <string name="app_name">calculator</string>

    <string name="history">History</string>
    <string name="calculator_mode">Calculator mode</string>

    <string name="mode_basic">Basic</string>
    <string name="mode_scientific">Scientific</string>
    <string name="mode_convert">Convert</string>

    <string name="error">Error</string>

</resources>
```

---

# 6.2 — Basic Calculator UI

## File: `app/src/main/res/drawable/ic_history.xml`

```xml
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
```

---

## File: `app/src/main/res/drawable/ic_calculator_mode.xml`

```xml
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
```

---

## File: `app/src/main/res/layout/activity_main.xml`

```xml
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
```

> Giữ `android:id="@+id/main"` vì `MainActivity` mặc định có thể dùng `findViewById(R.id.main)` để xử lý system bar insets.

---

## File: `app/src/main/res/layout/layout_basic_calculator.xml`

```xml
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

    <!-- TOP BAR -->
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

    <!-- DISPLAY -->
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

    <!-- KEYPAD -->
    <LinearLayout
        android:id="@+id/keypad"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:orientation="vertical">

        <!-- ROW 1 -->
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

        <!-- ROW 2 -->
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

        <!-- ROW 3 -->
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

        <!-- ROW 4 -->
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

        <!-- ROW 5 -->
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
```
## File: `app/src/main/java/com/example/calculator/MainActivity.java`

Ở giai đoạn 6.2 **không thêm calculator logic**. Giữ file template đang build tốt.

Điểm bắt buộc cần có:

```java
setContentView(R.layout.activity_main);
```

Nếu template có xử lý Edge-to-Edge/system bars thì giữ nguyên.

Ví dụ tham khảo:

```java
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
```
---
