# Refactor R2B — Nối `BasicCalculatorController` vào `MainActivity`

R2A đã tạo:

```text
app/src/main/java/com/example/calculator/ui/controller/
BasicCalculatorController.java
```

R2B sẽ:

```text
MainActivity
    ↓
khởi tạo BasicCalculatorController
    ↓
basicController.setup()
    ↓
toàn bộ Basic listener/render/history
không còn nằm trong MainActivity
```

Sau bước này, cả hai calculator mode đều được quản lý bởi controller riêng:

```text
Basic
→ BasicCalculatorController

Scientific
→ ScientificCalculatorController
```

---

# 1. Phạm vi thay đổi

Ở bước này chỉ thay:

```text
app/src/main/java/com/example/calculator/MainActivity.java
```

Không sửa:

```text
BasicCalculatorController.java
ScientificCalculatorController.java

CalculatorEngine.java
CalculatorState.java
ScientificEngine.java

HistoryManager.java
HistoryBottomSheet.java

activity_main.xml
layout_basic_calculator.xml
layout_scientific_calculator.xml
popup_calculator_mode.xml
```

---

# 2. Thay toàn bộ `MainActivity.java`

File:

```text
app/src/main/java/com/example/calculator/MainActivity.java
```

Thay toàn bộ bằng:

```java
package com.example.calculator;

import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.PopupWindow;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.example.calculator.engine.CalculatorEngine;
import com.example.calculator.engine.CalculatorMode;
import com.example.calculator.engine.ScientificEngine;
import com.example.calculator.history.HistoryManager;
import com.example.calculator.ui.HistoryBottomSheet;
import com.example.calculator.ui.controller.BasicCalculatorController;
import com.example.calculator.ui.controller.ScientificCalculatorController;

/**
 * Main screen của Calculator.
 *
 * MainActivity chỉ chịu trách nhiệm:
 *
 * - Lifecycle
 * - System bars
 * - Bind ba mode root
 * - Khởi tạo controller
 * - Mở History
 * - Mở Mode popup
 * - Chuyển Basic / Scientific / Convert
 *
 * Logic và interaction của từng calculator nằm trong:
 *
 * - BasicCalculatorController
 * - ScientificCalculatorController
 */
public class MainActivity
        extends AppCompatActivity {

    // =========================================================
    // CONSTANTS
    // =========================================================

    private static final String TAG_HISTORY_BOTTOM_SHEET =
            "HistoryBottomSheet";


    // =========================================================
    // SHARED DATA
    // =========================================================

    private HistoryManager historyManager;


    // =========================================================
    // CONTROLLERS
    // =========================================================

    private BasicCalculatorController
            basicCalculatorController;


    private ScientificCalculatorController
            scientificCalculatorController;


    // =========================================================
    // MODE
    // =========================================================

    private CalculatorMode currentMode;

    private PopupWindow modePopupWindow;


    // =========================================================
    // MODE ROOT VIEWS
    // =========================================================

    private View basicCalculatorLayout;

    private View scientificCalculatorLayout;

    private View converterLayout;


    // =========================================================
    // LIFECYCLE
    // =========================================================

    @Override
    protected void onCreate(
            Bundle savedInstanceState
    ) {

        super.onCreate(
                savedInstanceState
        );


        EdgeToEdge.enable(this);


        setContentView(
                R.layout.activity_main
        );


        setupSystemBars();


        initializeSharedData();


        bindModeLayouts();


        setupControllers();


        setupHistoryButton();


        setupModeButton();


        renderCalculatorMode();
    }


    // =========================================================
    // SYSTEM BARS
    // =========================================================

    /**
     * Giữ cách xử lý Edge-to-Edge của Empty Views Activity.
     */
    private void setupSystemBars() {

        ViewCompat.setOnApplyWindowInsetsListener(

                findViewById(
                        R.id.main
                ),

                (view, insets) -> {

                    Insets systemBars =
                            insets.getInsets(
                                    WindowInsetsCompat
                                            .Type
                                            .systemBars()
                            );


                    view.setPadding(

                            systemBars.left,

                            systemBars.top,

                            systemBars.right,

                            systemBars.bottom
                    );


                    return insets;
                }
        );
    }


    // =========================================================
    // INITIALIZATION
    // =========================================================

    private void initializeSharedData() {

        historyManager =
                new HistoryManager(
                        getApplicationContext()
                );


        currentMode =
                CalculatorMode.BASIC;
    }


    // =========================================================
    // MODE ROOT BINDING
    // =========================================================

    private void bindModeLayouts() {

        basicCalculatorLayout =
                findViewById(
                        R.id.basicCalculatorLayout
                );


        scientificCalculatorLayout =
                findViewById(
                        R.id.scientificCalculatorLayout
                );


        converterLayout =
                findViewById(
                        R.id.converterLayout
                );
    }


    // =========================================================
    // CONTROLLERS
    // =========================================================

    private void setupControllers() {

        setupBasicController();


        setupScientificController();
    }


    private void setupBasicController() {

        basicCalculatorController =
                new BasicCalculatorController(

                        basicCalculatorLayout,

                        new CalculatorEngine(),

                        historyManager
                );


        basicCalculatorController.setup();
    }


    private void setupScientificController() {

        scientificCalculatorController =
                new ScientificCalculatorController(

                        scientificCalculatorLayout,

                        new ScientificEngine(),

                        historyManager
                );


        scientificCalculatorController.setup();
    }


    // =========================================================
    // HISTORY
    // =========================================================

    private void setupHistoryButton() {

        findViewById(
                R.id.btnHistory
        ).setOnClickListener(view -> {

            /*
             * Không mở nhiều History Bottom Sheet chồng nhau.
             */
            if (getSupportFragmentManager()
                    .findFragmentByTag(
                            TAG_HISTORY_BOTTOM_SHEET
                    ) != null) {

                return;
            }


            HistoryBottomSheet historyBottomSheet =
                    new HistoryBottomSheet();


            historyBottomSheet.show(

                    getSupportFragmentManager(),

                    TAG_HISTORY_BOTTOM_SHEET
            );
        });
    }


    // =========================================================
    // MODE BUTTON
    // =========================================================

    private void setupModeButton() {

        findViewById(
                R.id.btnMode
        ).setOnClickListener(view -> {

            /*
             * Tap lại trong khi popup đang mở:
             *
             * → đóng popup.
             */
            if (modePopupWindow != null
                    && modePopupWindow.isShowing()) {

                modePopupWindow.dismiss();

                return;
            }


            showModePopup(
                    view
            );
        });
    }


    // =========================================================
    // MODE POPUP
    // =========================================================

    private void showModePopup(
            View anchor
    ) {

        View popupView =
                LayoutInflater
                        .from(this)
                        .inflate(
                                R.layout.popup_calculator_mode,
                                null,
                                false
                        );


        modePopupWindow =
                new PopupWindow(

                        popupView,

                        ViewGroup.LayoutParams.WRAP_CONTENT,

                        ViewGroup.LayoutParams.WRAP_CONTENT,

                        true
                );


        modePopupWindow.setBackgroundDrawable(

                new ColorDrawable(
                        Color.TRANSPARENT
                )
        );


        modePopupWindow.setOutsideTouchable(
                true
        );


        modePopupWindow.setElevation(

                getResources()
                        .getDimension(
                                R.dimen.spacing_8
                        )
        );


        updateModeCheckmarks(
                popupView
        );


        setupModePopupActions(
                popupView
        );


        modePopupWindow.setOnDismissListener(
                () -> modePopupWindow = null
        );


        /*
         * Measure popup để căn cạnh phải popup
         * với cạnh phải của btnMode.
         */
        popupView.measure(

                View.MeasureSpec.makeMeasureSpec(
                        0,
                        View.MeasureSpec.UNSPECIFIED
                ),

                View.MeasureSpec.makeMeasureSpec(
                        0,
                        View.MeasureSpec.UNSPECIFIED
                )
        );


        int popupWidth =
                popupView.getMeasuredWidth();


        int xOffset =
                anchor.getWidth()
                        - popupWidth;


        int yOffset =
                getResources()
                        .getDimensionPixelSize(
                                R.dimen.spacing_8
                        );


        modePopupWindow.showAsDropDown(

                anchor,

                xOffset,

                yOffset
        );
    }


    private void updateModeCheckmarks(
            View popupView
    ) {

        ImageView checkBasic =
                popupView.findViewById(
                        R.id.checkModeBasic
                );


        ImageView checkScientific =
                popupView.findViewById(
                        R.id.checkModeScientific
                );


        ImageView checkConvert =
                popupView.findViewById(
                        R.id.checkModeConvert
                );


        checkBasic.setVisibility(

                isCurrentMode(
                        CalculatorMode.BASIC
                )

                        ? View.VISIBLE
                        : View.GONE
        );


        checkScientific.setVisibility(

                isCurrentMode(
                        CalculatorMode.SCIENTIFIC
                )

                        ? View.VISIBLE
                        : View.GONE
        );


        checkConvert.setVisibility(

                isCurrentMode(
                        CalculatorMode.CONVERT
                )

                        ? View.VISIBLE
                        : View.GONE
        );
    }


    private void setupModePopupActions(
            View popupView
    ) {

        popupView
                .findViewById(
                        R.id.rowModeBasic
                )
                .setOnClickListener(view -> {

                    selectCalculatorMode(
                            CalculatorMode.BASIC
                    );
                });


        popupView
                .findViewById(
                        R.id.rowModeScientific
                )
                .setOnClickListener(view -> {

                    selectCalculatorMode(
                            CalculatorMode.SCIENTIFIC
                    );
                });


        popupView
                .findViewById(
                        R.id.rowModeConvert
                )
                .setOnClickListener(view -> {

                    selectCalculatorMode(
                            CalculatorMode.CONVERT
                    );
                });
    }


    private void selectCalculatorMode(
            CalculatorMode mode
    ) {

        setCalculatorMode(
                mode
        );


        if (modePopupWindow != null) {

            modePopupWindow.dismiss();
        }
    }


    private boolean isCurrentMode(
            CalculatorMode mode
    ) {

        return currentMode == mode;
    }


    // =========================================================
    // MODE SWITCHING
    // =========================================================

    private void setCalculatorMode(
            CalculatorMode mode
    ) {

        if (mode == null) {

            return;
        }


        currentMode =
                mode;


        renderCalculatorMode();
    }


    private void renderCalculatorMode() {

        basicCalculatorLayout.setVisibility(

                currentMode == CalculatorMode.BASIC

                        ? View.VISIBLE
                        : View.GONE
        );


        scientificCalculatorLayout.setVisibility(

                currentMode == CalculatorMode.SCIENTIFIC

                        ? View.VISIBLE
                        : View.GONE
        );


        converterLayout.setVisibility(

                currentMode == CalculatorMode.CONVERT

                        ? View.VISIBLE
                        : View.GONE
        );
    }
}
```

---

# 3. Các phần Basic đã bị loại khỏi `MainActivity`

Sau khi thay toàn bộ file, `MainActivity` không còn các field:

```text
CalculatorEngine calculatorEngine

TextView tvExpression
TextView tvResult
```

Không còn các method:

```text
setupDigitButtons()
bindDigitButton()

setupOperatorButtons()
bindOperatorButton()

setupFunctionButtons()

calculateAndSaveHistory()

renderState()
```

Các phần trên hiện nằm trong:

```text
BasicCalculatorController.java
```

`MainActivity` chỉ còn:

```java
private BasicCalculatorController
        basicCalculatorController;
```

và:

```java
private void setupBasicController() {

    basicCalculatorController =
            new BasicCalculatorController(
                    basicCalculatorLayout,
                    new CalculatorEngine(),
                    historyManager
            );

    basicCalculatorController.setup();
}
```

---

# 4. Cấu trúc `MainActivity` sau R2B

```text
MainActivity
│
├── onCreate()
├── setupSystemBars()
├── initializeSharedData()
├── bindModeLayouts()
│
├── setupControllers()
│   ├── setupBasicController()
│   └── setupScientificController()
│
├── setupHistoryButton()
│
├── setupModeButton()
├── showModePopup()
├── updateModeCheckmarks()
├── setupModePopupActions()
├── selectCalculatorMode()
├── isCurrentMode()
│
├── setCalculatorMode()
└── renderCalculatorMode()
```

Không còn:

```text
Basic button mapping
Basic render
Basic History save logic

Scientific button mapping
Scientific state
Scientific render
Scientific History save logic
```

---

# 5. Build checkpoint

Chạy:

```text
Build
→ Make Project
```

Checklist:

```text
[ ] MainActivity.java compile

[ ] BasicCalculatorController import resolve

[ ] ScientificCalculatorController import resolve

[ ] CalculatorEngine import resolve

[ ] ScientificEngine import resolve

[ ] Không còn import thừa:
    android.widget.Button
    android.widget.TextView
    CalculatorState

[ ] Không còn unresolved method:
    setupDigitButtons
    setupOperatorButtons
    setupFunctionButtons
    calculateAndSaveHistory
    renderState

[ ] Không có duplicate listener

[ ] History compile

[ ] Mode popup compile
```

---

# 6. Lỗi thường gặp

## Lỗi 1 — Không tìm thấy `BasicCalculatorController`

Thông báo:

```text
Cannot resolve symbol BasicCalculatorController
```

Kiểm tra file:

```text
app/src/main/java/com/example/calculator/ui/controller/
BasicCalculatorController.java
```

Dòng package phải là:

```java
package com.example.calculator.ui.controller;
```

Import trong `MainActivity`:

```java
import com.example.calculator.ui.controller.BasicCalculatorController;
```

---

## Lỗi 2 — Basic root bị null

Thông báo có thể là:

```text
IllegalArgumentException:
Basic root view cannot be null.
```

Kiểm tra `activity_main.xml` có:

```xml
<include
    android:id="@+id/basicCalculatorLayout"
    layout="@layout/layout_basic_calculator"
    ... />
```

ID phải đúng:

```text
basicCalculatorLayout
```

---

## Lỗi 3 — Không tìm thấy button Basic

Ví dụ:

```text
NullPointerException
khi gọi setOnClickListener
```

Kiểm tra `layout_basic_calculator.xml` có đúng các ID:

```text
btn0
btn1
btn2
btn3
btn4
btn5
btn6
btn7
btn8
btn9

btnAdd
btnSubtract
btnMultiply
btnDivide

btnDecimal
btnEquals
btnClear
btnBackspace
btnToggleSign
btnPercent
```

Nếu XML dùng ID khác, sửa mapping trong:

```text
BasicCalculatorController.java
```

Không sửa ngẫu nhiên nhiều file cùng lúc.

---

## Lỗi 4 — `HistoryManager` bị null

Bản R2B khởi tạo theo thứ tự:

```text
initializeSharedData()
        ↓
historyManager tồn tại
        ↓
setupControllers()
```

Không đổi thành:

```text
setupControllers()
        ↓
initializeSharedData()
```

Nếu đổi sai thứ tự, hai controller vẫn chạy nhưng không lưu được History.

---

# 7. Runtime regression test

Sau khi Build pass, chạy app.

## Test A — Basic input

```text
7 8

Expected:
78
```

```text
1 . 5

Expected:
1.5
```

```text
1 . 5 . 2

Expected:
1.52
```

---

## Test B — Basic calculation

Nhấn `AC` trước mỗi test:

```text
2 + 3 =
Expected: 5
```

```text
10 − 2 =
Expected: 8
```

```text
5 × 8 =
Expected: 40
```

```text
20 ÷ 4 =
Expected: 5
```

```text
0.1 + 0.2 =
Expected: 0.3
```

---

## Test C — Basic utility

```text
123
⌫

Expected:
12
```

```text
25
±

Expected:
-25
```

```text
50
%

Expected:
0.5
```

```text
5 ÷ 0 =

Expected:
Error
```

---

## Test D — Basic History

Thực hiện:

```text
25 × 4 =
```

Mở:

```text
History
```

Expected:

```text
25 × 4
100
```

Nhấn `=` lần nữa không được tạo thêm item trùng.

Phép:

```text
5 ÷ 0 =
```

không được lưu vào History.

---

## Test E — Scientific regression

Chuyển:

```text
Mode
→ Scientific
```

Test:

```text
16
√x

Expected:
4
```

```text
2
x³

Expected:
8
```

```text
2
xʸ
3
=

Expected:
8
```

```text
30
sin

DEG Expected:
0.5
```

---

## Test F — Mode regression

Chuyển liên tục:

```text
Basic
→ Scientific
→ Convert
→ Basic
```

Expected:

```text
Không crash
Top bar luôn hiển thị
Popup có đúng một checkmark
Basic state vẫn còn khi quay lại
Scientific state vẫn còn khi quay lại
```

---

# 8. Kết quả sau R2B

```text
MainActivity
│
├── Lifecycle
├── System bars
├── Shared History launcher
├── Mode popup
├── Mode switching
├── Basic controller initialization
└── Scientific controller initialization


BasicCalculatorController
│
├── Basic button listeners
├── CalculatorEngine
├── CalculatorState render
└── Basic History


ScientificCalculatorController
│
├── Scientific button listeners
├── Scientific state
├── ScientificEngine
├── Scientific render
└── Scientific History
```

Trạng thái:

```text
[✓] R1A
    Tạo ScientificCalculatorController

[~] R1B
    Nối Scientific controller
    Chờ xác nhận Build/Run

[✓] R2A
    Tạo BasicCalculatorController

[~] R2B
    Nối Basic controller
    Xóa Basic code khỏi MainActivity
    Chờ Build/Run

NEXT

R3A
→ tạo ModeSelectorPopup
→ chưa sửa MainActivity
→ Build
```
