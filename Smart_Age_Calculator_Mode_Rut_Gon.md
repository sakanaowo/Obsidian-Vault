# Smart Age Calculator Mode — Bản đầy đủ rút gọn

Dưới đây là bản **đầy đủ nhưng rút gọn**, chỉ giữ các thành phần cần thiết:

```text
CalculatorMode
SmartAgeEngine
SmartAgeController
Smart Age layout
MainActivity integration
Mode popup integration
```

Phần tính tuổi dùng `Calendar`, không cần thay đổi Gradle hoặc yêu cầu Android API 26.

---

# 1. Thêm mode `SMART_AGE`

File:

```text
engine/CalculatorMode.java
```

```java
package com.example.calculator.engine;

public enum CalculatorMode {

    BASIC,

    SCIENTIFIC,

    CONVERT,

    SMART_AGE
}
```

---

# 2. Tạo `SmartAgeEngine.java`

File:

```text
app/src/main/java/com/example/calculator/engine/
SmartAgeEngine.java
```

```java
package com.example.calculator.engine;

import java.util.Calendar;

/**
 * Tính tuổi thật và tạo kết quả vui.
 *
 * Phần tuổi thật được tính chính xác từ ngày sinh.
 * Smart Age chỉ mang tính giải trí.
 */
public final class SmartAgeEngine {

    private static final String[] JOKES = {

            "Bạn không già đi, chỉ tăng version.",

            "Thanh xuân chưa mất, chỉ đang chạy nền.",

            "Tuổi chỉ là con số. Deadline mới là vấn đề.",

            "Kinh nghiệm đang tăng. Pin thì chưa chắc.",

            "Bạn vẫn trẻ cho đến khi mở camera trước.",

            "Phiên bản hiện tại khá ổn định, chưa cần cập nhật."
    };


    public AgeResult calculate(
            int birthYear,
            int birthMonth,
            int birthDay
    ) {

        Calendar today =
                Calendar.getInstance();

        clearTime(
                today
        );


        Calendar birthDate =
                Calendar.getInstance();

        birthDate.setLenient(
                false
        );

        birthDate.set(
                birthYear,
                birthMonth,
                birthDay,
                0,
                0,
                0
        );

        birthDate.set(
                Calendar.MILLISECOND,
                0
        );


        try {

            birthDate.getTime();

        } catch (IllegalArgumentException exception) {

            return AgeResult.error(
                    "Invalid birth date."
            );
        }


        if (birthDate.after(today)) {

            return AgeResult.error(
                    "Birth date cannot be in the future."
            );
        }


        /*
         * Tính số năm hoàn chỉnh.
         */
        int years =
                today.get(Calendar.YEAR)

                        - birthDate.get(
                        Calendar.YEAR
                );


        Calendar cursor =
                (Calendar) birthDate.clone();

        cursor.add(
                Calendar.YEAR,
                years
        );


        if (cursor.after(today)) {

            years--;

            cursor =
                    (Calendar) birthDate.clone();

            cursor.add(
                    Calendar.YEAR,
                    years
            );
        }


        /*
         * Tính số tháng còn lại.
         */
        int months =
                0;


        while (true) {

            Calendar nextMonth =
                    (Calendar) cursor.clone();

            nextMonth.add(
                    Calendar.MONTH,
                    1
            );


            if (nextMonth.after(today)) {

                break;
            }


            cursor =
                    nextMonth;

            months++;
        }


        /*
         * Tính số ngày còn lại.
         */
        int days =
                0;


        while (true) {

            Calendar nextDay =
                    (Calendar) cursor.clone();

            nextDay.add(
                    Calendar.DAY_OF_MONTH,
                    1
            );


            if (nextDay.after(today)) {

                break;
            }


            cursor =
                    nextDay;

            days++;
        }


        int daysUntilBirthday =
                calculateDaysUntilBirthday(
                        birthDate,
                        today
                );


        /*
         * Kết quả joke ổn định theo ngày sinh.
         * Không thay đổi mỗi lần render.
         */
        int seed =
                birthYear * 31

                        + birthMonth * 17

                        + birthDay;


        int smartAge =
                Math.max(

                        5,

                        years - 3
                                + Math.floorMod(
                                seed,
                                7
                        )
                );


        String joke =
                JOKES[
                        Math.floorMod(
                                seed,
                                JOKES.length
                        )
                        ];


        return AgeResult.success(

                years,

                months,

                days,

                smartAge,

                daysUntilBirthday,

                joke
        );
    }


    private int calculateDaysUntilBirthday(
            Calendar birthDate,
            Calendar today
    ) {

        Calendar nextBirthday =
                (Calendar) birthDate.clone();


        nextBirthday.set(
                Calendar.YEAR,
                today.get(
                        Calendar.YEAR
                )
        );


        if (nextBirthday.before(today)) {

            nextBirthday.add(
                    Calendar.YEAR,
                    1
            );
        }


        int days =
                0;


        Calendar cursor =
                (Calendar) today.clone();


        while (cursor.before(nextBirthday)) {

            cursor.add(
                    Calendar.DAY_OF_MONTH,
                    1
            );

            days++;
        }


        return days;
    }


    private void clearTime(
            Calendar calendar
    ) {

        calendar.set(
                Calendar.HOUR_OF_DAY,
                0
        );

        calendar.set(
                Calendar.MINUTE,
                0
        );

        calendar.set(
                Calendar.SECOND,
                0
        );

        calendar.set(
                Calendar.MILLISECOND,
                0
        );
    }


    // =========================================================
    // RESULT
    // =========================================================

    public static final class AgeResult {

        private final boolean success;

        private final String errorMessage;

        private final int years;

        private final int months;

        private final int days;

        private final int smartAge;

        private final int daysUntilBirthday;

        private final String joke;


        private AgeResult(
                boolean success,
                String errorMessage,
                int years,
                int months,
                int days,
                int smartAge,
                int daysUntilBirthday,
                String joke
        ) {

            this.success =
                    success;

            this.errorMessage =
                    errorMessage;

            this.years =
                    years;

            this.months =
                    months;

            this.days =
                    days;

            this.smartAge =
                    smartAge;

            this.daysUntilBirthday =
                    daysUntilBirthday;

            this.joke =
                    joke;
        }


        public static AgeResult success(
                int years,
                int months,
                int days,
                int smartAge,
                int daysUntilBirthday,
                String joke
        ) {

            return new AgeResult(

                    true,

                    "",

                    years,

                    months,

                    days,

                    smartAge,

                    daysUntilBirthday,

                    joke
            );
        }


        public static AgeResult error(
                String message
        ) {

            return new AgeResult(

                    false,

                    message,

                    0,

                    0,

                    0,

                    0,

                    0,

                    ""
            );
        }


        public boolean isSuccess() {
            return success;
        }

        public String getErrorMessage() {
            return errorMessage;
        }

        public int getYears() {
            return years;
        }

        public int getMonths() {
            return months;
        }

        public int getDays() {
            return days;
        }

        public int getSmartAge() {
            return smartAge;
        }

        public int getDaysUntilBirthday() {
            return daysUntilBirthday;
        }

        public String getJoke() {
            return joke;
        }
    }
}
```

Engine rút gọn chỉ còn:

```text
calculate()
calculateDaysUntilBirthday()
clearTime()
AgeResult
```

---

# 3. Thêm strings

File:

```text
res/values/strings.xml
```

Thêm:

```xml
<!-- Smart Age -->
<string name="mode_smart_age">Smart Age</string>

<string name="smart_age_title">
    Smart Age Calculator
</string>

<string name="smart_age_description">
    Real age is accurate. Smart age is completely made up.
</string>

<string name="smart_age_select_date">
    Select birth date
</string>

<string name="smart_age_no_date">
    No birth date selected
</string>

<string name="smart_age_real_format">
    %1$d years old
</string>

<string name="smart_age_detail_format">
    %1$d years, %2$d months, %3$d days
</string>

<string name="smart_age_joke_format">
    Smart age: %1$d
</string>

<string name="smart_age_birthday_today">
    Your birthday is today 🎂
</string>

<string name="smart_age_birthday_format">
    Next birthday in %1$d days
</string>
```

---

# 4. Tạo layout Smart Age

File:

```text
res/layout/layout_smart_age_calculator.xml
```

```xml
<?xml version="1.0" encoding="utf-8"?>

<ScrollView
    xmlns:android="http://schemas.android.com/apk/res/android"

    android:id="@+id/smartAgeCalculatorRoot"

    android:layout_width="match_parent"
    android:layout_height="match_parent"

    android:background="@color/calculator_background"

    android:fillViewport="true">


    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"

        android:orientation="vertical"

        android:paddingStart="@dimen/screen_padding_horizontal"
        android:paddingEnd="@dimen/screen_padding_horizontal"

        android:paddingTop="@dimen/spacing_24"
        android:paddingBottom="@dimen/spacing_24">


        <TextView
            android:layout_width="match_parent"
            android:layout_height="wrap_content"

            android:gravity="center"

            android:text="@string/smart_age_title"

            android:textColor="@color/calculator_text_primary"

            android:textSize="28sp"

            android:textStyle="bold" />


        <TextView
            android:layout_width="match_parent"
            android:layout_height="wrap_content"

            android:layout_marginTop="@dimen/spacing_8"

            android:gravity="center"

            android:text="@string/smart_age_description"

            android:textColor="@color/calculator_text_secondary"

            android:textSize="15sp" />


        <Button
            android:id="@+id/btnSelectBirthDate"

            android:layout_width="match_parent"
            android:layout_height="@dimen/unit_picker_confirm_height"

            android:layout_marginTop="@dimen/spacing_24"

            android:background="@drawable/bg_unit_confirm"

            android:text="@string/smart_age_select_date"

            android:textAllCaps="false"

            android:textColor="@color/calculator_text_primary"

            android:textSize="@dimen/text_unit_picker_confirm"

            android:textStyle="bold" />


        <TextView
            android:id="@+id/tvSmartBirthDate"

            android:layout_width="match_parent"
            android:layout_height="wrap_content"

            android:layout_marginTop="@dimen/spacing_16"

            android:gravity="center"

            android:text="@string/smart_age_no_date"

            android:textColor="@color/calculator_text_secondary"

            android:textSize="17sp" />


        <LinearLayout
            android:id="@+id/smartAgeResultContainer"

            android:layout_width="match_parent"
            android:layout_height="wrap_content"

            android:layout_marginTop="@dimen/spacing_24"

            android:background="@drawable/bg_mode_popup"

            android:orientation="vertical"

            android:padding="@dimen/spacing_20"

            android:visibility="gone">


            <TextView
                android:id="@+id/tvSmartRealAge"

                android:layout_width="match_parent"
                android:layout_height="wrap_content"

                android:gravity="center"

                android:textColor="@color/calculator_text_primary"

                android:textSize="30sp"

                android:textStyle="bold" />


            <TextView
                android:id="@+id/tvSmartAgeDetails"

                android:layout_width="match_parent"
                android:layout_height="wrap_content"

                android:layout_marginTop="@dimen/spacing_8"

                android:gravity="center"

                android:textColor="@color/calculator_text_secondary"

                android:textSize="17sp" />


            <TextView
                android:id="@+id/tvSmartJokeAge"

                android:layout_width="match_parent"
                android:layout_height="wrap_content"

                android:layout_marginTop="@dimen/spacing_20"

                android:gravity="center"

                android:textColor="@color/calculator_button_operator"

                android:textSize="22sp"

                android:textStyle="bold" />


            <TextView
                android:id="@+id/tvSmartBirthday"

                android:layout_width="match_parent"
                android:layout_height="wrap_content"

                android:layout_marginTop="@dimen/spacing_12"

                android:gravity="center"

                android:textColor="@color/calculator_text_primary"

                android:textSize="17sp" />


            <TextView
                android:id="@+id/tvSmartComment"

                android:layout_width="match_parent"
                android:layout_height="wrap_content"

                android:layout_marginTop="@dimen/spacing_20"

                android:gravity="center"

                android:textColor="@color/calculator_text_secondary"

                android:textSize="18sp"

                android:textStyle="italic" />

        </LinearLayout>

    </LinearLayout>

</ScrollView>
```

---

# 5. Tạo `SmartAgeController.java`

File:

```text
app/src/main/java/com/example/calculator/ui/controller/
SmartAgeController.java
```

```java
package com.example.calculator.ui.controller;

import android.app.DatePickerDialog;
import android.view.View;
import android.widget.TextView;

import com.example.calculator.R;
import com.example.calculator.engine.SmartAgeEngine;

import java.util.Calendar;
import java.util.Locale;

/**
 * Điều phối giao diện Smart Age.
 */
public final class SmartAgeController {

    private final View rootView;

    private final SmartAgeEngine engine;


    private View resultContainer;

    private TextView tvBirthDate;

    private TextView tvRealAge;

    private TextView tvAgeDetails;

    private TextView tvJokeAge;

    private TextView tvBirthday;

    private TextView tvComment;


    public SmartAgeController(
            View rootView,
            SmartAgeEngine engine
    ) {

        if (rootView == null
                || engine == null) {

            throw new IllegalArgumentException(
                    "Smart Age dependencies cannot be null."
            );
        }


        this.rootView =
                rootView;

        this.engine =
                engine;
    }


    public void setup() {

        bindViews();


        rootView
                .findViewById(
                        R.id.btnSelectBirthDate
                )
                .setOnClickListener(view -> {

                    showDatePicker();
                });
    }


    private void bindViews() {

        resultContainer =
                rootView.findViewById(
                        R.id.smartAgeResultContainer
                );


        tvBirthDate =
                rootView.findViewById(
                        R.id.tvSmartBirthDate
                );


        tvRealAge =
                rootView.findViewById(
                        R.id.tvSmartRealAge
                );


        tvAgeDetails =
                rootView.findViewById(
                        R.id.tvSmartAgeDetails
                );


        tvJokeAge =
                rootView.findViewById(
                        R.id.tvSmartJokeAge
                );


        tvBirthday =
                rootView.findViewById(
                        R.id.tvSmartBirthday
                );


        tvComment =
                rootView.findViewById(
                        R.id.tvSmartComment
                );
    }


    private void showDatePicker() {

        Calendar initialDate =
                Calendar.getInstance();


        initialDate.add(
                Calendar.YEAR,
                -18
        );


        DatePickerDialog dialog =
                new DatePickerDialog(

                        rootView.getContext(),

                        (picker,
                         year,
                         month,
                         day) -> {

                            calculateAndRender(

                                    year,

                                    month,

                                    day
                            );
                        },

                        initialDate.get(
                                Calendar.YEAR
                        ),

                        initialDate.get(
                                Calendar.MONTH
                        ),

                        initialDate.get(
                                Calendar.DAY_OF_MONTH
                        )
                );


        /*
         * Không cho chọn ngày trong tương lai.
         */
        dialog
                .getDatePicker()
                .setMaxDate(
                        System.currentTimeMillis()
                );


        dialog.show();
    }


    private void calculateAndRender(
            int year,
            int month,
            int day
    ) {

        SmartAgeEngine.AgeResult result =
                engine.calculate(

                        year,

                        month,

                        day
                );


        if (!result.isSuccess()) {

            resultContainer.setVisibility(
                    View.GONE
            );


            tvBirthDate.setText(
                    result.getErrorMessage()
            );


            return;
        }


        tvBirthDate.setText(

                String.format(

                        Locale.getDefault(),

                        "%02d/%02d/%04d",

                        day,

                        month + 1,

                        year
                )
        );


        tvRealAge.setText(

                rootView
                        .getContext()
                        .getString(

                                R.string.smart_age_real_format,

                                result.getYears()
                        )
        );


        tvAgeDetails.setText(

                rootView
                        .getContext()
                        .getString(

                                R.string.smart_age_detail_format,

                                result.getYears(),

                                result.getMonths(),

                                result.getDays()
                        )
        );


        tvJokeAge.setText(

                rootView
                        .getContext()
                        .getString(

                                R.string.smart_age_joke_format,

                                result.getSmartAge()
                        )
        );


        if (result.getDaysUntilBirthday() == 0) {

            tvBirthday.setText(
                    R.string.smart_age_birthday_today
            );

        } else {

            tvBirthday.setText(

                    rootView
                            .getContext()
                            .getString(

                                    R.string.smart_age_birthday_format,

                                    result.getDaysUntilBirthday()
                            )
            );
        }


        tvComment.setText(

                "“"

                        + result.getJoke()

                        + "”"
        );


        resultContainer.setVisibility(
                View.VISIBLE
        );
    }
}
```

Controller chỉ còn:

```text
setup()
bindViews()
showDatePicker()
calculateAndRender()
```

---

# 6. Include trong `activity_main.xml`

Trong:

```xml
<FrameLayout
    android:id="@+id/modeContentContainer"
    ...>
```

thêm:

```xml
<!-- SMART AGE -->
<include
    android:id="@+id/smartAgeCalculatorLayout"

    layout="@layout/layout_smart_age_calculator"

    android:layout_width="match_parent"
    android:layout_height="match_parent"

    android:visibility="gone" />
```

---

# 7. Nối vào `MainActivity`

## Imports

```java
import com.example.calculator.engine.SmartAgeEngine;
import com.example.calculator.ui.controller.SmartAgeController;
```

## Fields

```java
private View smartAgeCalculatorLayout;


private SmartAgeController
        smartAgeController;
```

## Bind layout

Trong method bind các mode:

```java
smartAgeCalculatorLayout =
        findViewById(
                R.id.smartAgeCalculatorLayout
        );
```

## Setup controller

Trong:

```java
private void setupControllers()
```

thêm:

```java
setupSmartAgeController();
```

Method mới:

```java
private void setupSmartAgeController() {

    smartAgeController =
            new SmartAgeController(

                    smartAgeCalculatorLayout,

                    new SmartAgeEngine()
            );


    smartAgeController.setup();
}
```

## Render mode

Trong:

```java
private void renderCalculatorMode()
```

thêm:

```java
smartAgeCalculatorLayout.setVisibility(

        currentMode == CalculatorMode.SMART_AGE

                ? View.VISIBLE
                : View.GONE
);
```

---

# 8. Thêm Smart Age vào Mode Popup

Trong:

```text
res/layout/popup_calculator_mode.xml
```

thêm một row:

```xml
<LinearLayout
    android:id="@+id/rowModeSmartAge"

    android:layout_width="match_parent"
    android:layout_height="@dimen/mode_row_height"

    android:background="@drawable/bg_mode_row"

    android:clickable="true"
    android:focusable="true"

    android:gravity="center_vertical"

    android:orientation="horizontal"

    android:paddingStart="@dimen/mode_row_horizontal_padding"
    android:paddingEnd="@dimen/mode_row_horizontal_padding">


    <ImageView
        android:id="@+id/checkModeSmartAge"

        android:layout_width="@dimen/mode_check_size"
        android:layout_height="@dimen/mode_check_size"

        android:src="@drawable/ic_check"

        android:visibility="gone" />


    <TextView
        android:layout_width="0dp"
        android:layout_height="wrap_content"

        android:layout_marginStart="@dimen/spacing_16"

        android:layout_weight="1"

        android:text="@string/mode_smart_age"

        android:textColor="@color/calculator_text_primary"

        android:textSize="@dimen/text_mode_item" />

</LinearLayout>
```

---

# 9. Cập nhật `ModeSelectorPopup`

Trong phần setup action:

```java
popupView
        .findViewById(
                R.id.rowModeSmartAge
        )
        .setOnClickListener(view -> {

            notifyModeSelected(
                    CalculatorMode.SMART_AGE
            );
        });
```

Trong phần update checkmark:

```java
ImageView checkSmartAge =
        popupView.findViewById(
                R.id.checkModeSmartAge
        );


checkSmartAge.setVisibility(

        currentMode == CalculatorMode.SMART_AGE

                ? View.VISIBLE
                : View.GONE
);
```

Tên `notifyModeSelected()` cần thay bằng tên callback thực tế đang được dùng trong `ModeSelectorPopup`.

---

# 10. Kết quả hoạt động

```text
Mode
  ↓
Smart Age
  ↓
Select birth date
  ↓
15/08/2003
  ↓
22 years old
22 years, 11 months, 14 days

Smart age: 24

Next birthday in 17 days

“Bạn không già đi, chỉ tăng version.”
```

Mode này không lưu vào History vì kết quả không phải calculation thông thường.

## Test tối thiểu

```text
Ngày tương lai
→ không chọn được

Sinh nhật hôm nay
→ Your birthday is today

Ngày hợp lệ
→ hiển thị năm, tháng, ngày

Đổi mode rồi quay lại
→ kết quả vẫn còn

Mở Mode popup
→ Smart Age có checkmark

Basic / Scientific / Convert
→ không bị ảnh hưởng
```
