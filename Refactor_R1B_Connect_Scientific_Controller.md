# Refactor R1B — Nối `ScientificCalculatorController` vào `MainActivity`

Bước R1A đã tạo:

```text
app/src/main/java/com/example/calculator/ui/controller/
ScientificCalculatorController.java
```

R1B sẽ:

```text
MainActivity
    ↓
khởi tạo ScientificCalculatorController
    ↓
controller.setup()
    ↓
toàn bộ Scientific listener/state/render
không còn nằm trong MainActivity
```

## Phạm vi thay đổi

Ở bước này:

```text
Chỉ thay toàn bộ MainActivity.java

Không sửa:
ScientificCalculatorController.java
ScientificEngine.java
CalculatorEngine.java
CalculatorState.java
HistoryManager.java
HistoryBottomSheet.java
XML layouts
```

Bản `MainActivity` bên dưới được tái dựng từ lịch sử project đến Bước 8.7 và giữ lại:

```text
Basic Calculator
History
Mode Popup
Mode switching
Scientific Controller
Convert placeholder
System bars
```

---

# 1. Thay toàn bộ `MainActivity.java`

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
import android.widget.Button;
import android.widget.ImageView;
import android.widget.PopupWindow;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.example.calculator.engine.CalculatorEngine;
import com.example.calculator.engine.CalculatorMode;
import com.example.calculator.engine.CalculatorState;
import com.example.calculator.engine.ScientificEngine;
import com.example.calculator.history.HistoryManager;
import com.example.calculator.ui.HistoryBottomSheet;
import com.example.calculator.ui.controller.ScientificCalculatorController;

/**
 * Main screen của Calculator.
 *
 * MainActivity chỉ chịu trách nhiệm:
 *
 * - Lifecycle
 * - System bars
 * - Basic Calculator UI
 * - History launcher
 * - Mode popup
 * - Chuyển Basic / Scientific / Convert
 * - Khởi tạo ScientificCalculatorController
 *
 * Scientific input state, listeners và render đã được chuyển sang:
 *
 * ScientificCalculatorController
 */
public class MainActivity
        extends AppCompatActivity {

    // =========================================================
    // CONSTANTS
    // =========================================================

    private static final String TAG_HISTORY_BOTTOM_SHEET =
            "HistoryBottomSheet";


    // =========================================================
    // BASIC CALCULATOR
    // =========================================================

    private CalculatorEngine calculatorEngine;

    private TextView tvExpression;

    private TextView tvResult;


    // =========================================================
    // SHARED DATA
    // =========================================================

    private HistoryManager historyManager;


    // =========================================================
    // SCIENTIFIC CONTROLLER
    // =========================================================

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


        // -----------------------------------------------------
        // ENGINES / DATA
        // -----------------------------------------------------

        calculatorEngine =
                new CalculatorEngine();


        historyManager =
                new HistoryManager(
                        getApplicationContext()
                );


        currentMode =
                CalculatorMode.BASIC;


        // -----------------------------------------------------
        // VIEW BINDING
        // -----------------------------------------------------

        bindViews();


        // -----------------------------------------------------
        // BASIC CALCULATOR
        // -----------------------------------------------------

        setupDigitButtons();

        setupOperatorButtons();

        setupFunctionButtons();


        // -----------------------------------------------------
        // SCIENTIFIC CALCULATOR
        // -----------------------------------------------------

        setupScientificController();


        // -----------------------------------------------------
        // SHARED TOP BAR
        // -----------------------------------------------------

        setupHistoryButton();

        setupModeButton();


        // -----------------------------------------------------
        // INITIAL RENDER
        // -----------------------------------------------------

        renderCalculatorMode();

        renderState();
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
    // BIND VIEWS
    // =========================================================

    private void bindViews() {

        // -----------------------------------------------------
        // MODE ROOTS
        // -----------------------------------------------------

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


        // -----------------------------------------------------
        // BASIC DISPLAY
        // -----------------------------------------------------

        tvExpression =
                basicCalculatorLayout.findViewById(
                        R.id.tvExpression
                );


        tvResult =
                basicCalculatorLayout.findViewById(
                        R.id.tvResult
                );
    }


    // =========================================================
    // SCIENTIFIC CONTROLLER
    // =========================================================

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
    // BASIC DIGIT BUTTONS
    // =========================================================

    private void setupDigitButtons() {

        bindDigitButton(
                R.id.btn0,
                "0"
        );

        bindDigitButton(
                R.id.btn1,
                "1"
        );

        bindDigitButton(
                R.id.btn2,
                "2"
        );

        bindDigitButton(
                R.id.btn3,
                "3"
        );

        bindDigitButton(
                R.id.btn4,
                "4"
        );

        bindDigitButton(
                R.id.btn5,
                "5"
        );

        bindDigitButton(
                R.id.btn6,
                "6"
        );

        bindDigitButton(
                R.id.btn7,
                "7"
        );

        bindDigitButton(
                R.id.btn8,
                "8"
        );

        bindDigitButton(
                R.id.btn9,
                "9"
        );
    }


    private void bindDigitButton(
            int buttonId,
            String digit
    ) {

        Button button =
                basicCalculatorLayout.findViewById(
                        buttonId
                );


        button.setOnClickListener(view -> {

            calculatorEngine.inputDigit(
                    digit
            );


            renderState();
        });
    }


    // =========================================================
    // BASIC OPERATOR BUTTONS
    // =========================================================

    private void setupOperatorButtons() {

        bindOperatorButton(
                R.id.btnAdd,
                "+"
        );

        bindOperatorButton(
                R.id.btnSubtract,
                "−"
        );

        bindOperatorButton(
                R.id.btnMultiply,
                "×"
        );

        bindOperatorButton(
                R.id.btnDivide,
                "÷"
        );
    }


    private void bindOperatorButton(
            int buttonId,
            String operator
    ) {

        Button button =
                basicCalculatorLayout.findViewById(
                        buttonId
                );


        button.setOnClickListener(view -> {

            calculatorEngine.inputOperator(
                    operator
            );


            renderState();
        });
    }


    // =========================================================
    // BASIC FUNCTION BUTTONS
    // =========================================================

    private void setupFunctionButtons() {

        // Decimal
        basicCalculatorLayout
                .findViewById(
                        R.id.btnDecimal
                )
                .setOnClickListener(view -> {

                    calculatorEngine.inputDecimal();

                    renderState();
                });


        // Equals
        basicCalculatorLayout
                .findViewById(
                        R.id.btnEquals
                )
                .setOnClickListener(view -> {

                    calculateAndSaveHistory();
                });


        // AC
        basicCalculatorLayout
                .findViewById(
                        R.id.btnClear
                )
                .setOnClickListener(view -> {

                    calculatorEngine.clear();

                    renderState();
                });


        // Backspace
        basicCalculatorLayout
                .findViewById(
                        R.id.btnBackspace
                )
                .setOnClickListener(view -> {

                    calculatorEngine.backspace();

                    renderState();
                });


        // Toggle sign
        basicCalculatorLayout
                .findViewById(
                        R.id.btnToggleSign
                )
                .setOnClickListener(view -> {

                    calculatorEngine.toggleSign();

                    renderState();
                });


        // Percent
        basicCalculatorLayout
                .findViewById(
                        R.id.btnPercent
                )
                .setOnClickListener(view -> {

                    calculatorEngine.percent();

                    renderState();
                });
    }


    // =========================================================
    // BASIC CALCULATE + HISTORY
    // =========================================================

    /**
     * Thực hiện phép tính Basic và chỉ lưu History nếu:
     *
     * - đang có phép tính hợp lệ;
     * - kết quả không Error;
     * - expression và result không rỗng.
     *
     * Việc kiểm tra state trước calculate() giúp tránh lưu trùng
     * khi nhấn "=" nhiều lần.
     */
    private void calculateAndSaveHistory() {

        CalculatorState beforeState =
                calculatorEngine.getState();


        boolean hasPendingCalculation =

                !beforeState.isError()

                        && beforeState.getFirstOperand() != null

                        && beforeState.getOperator() != null

                        && !beforeState.isWaitingForOperand();


        calculatorEngine.calculate();


        CalculatorState afterState =
                calculatorEngine.getState();


        if (hasPendingCalculation

                && !afterState.isError()

                && afterState.getExpression() != null

                && !afterState
                        .getExpression()
                        .trim()
                        .isEmpty()

                && afterState.getResult() != null

                && !afterState
                        .getResult()
                        .trim()
                        .isEmpty()) {


            historyManager.add(

                    afterState.getExpression(),

                    afterState.getResult()
            );
        }


        renderState();
    }


    // =========================================================
    // BASIC RENDER
    // =========================================================

    private void renderState() {

        CalculatorState state =
                calculatorEngine.getState();


        String expression =
                state.getExpression();


        String result =
                state.getResult();


        tvExpression.setText(

                expression == null

                        ? ""
                        : expression
        );


        tvResult.setText(

                result == null

                        ? "0"
                        : result
        );
    }


    // =========================================================
    // HISTORY
    // =========================================================

    private void setupHistoryButton() {

        findViewById(
                R.id.btnHistory
        ).setOnClickListener(view -> {

            /*
             * Không mở nhiều Bottom Sheet chồng nhau.
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

# 2. Những phần Scientific đã bị loại khỏi `MainActivity`

Sau khi thay toàn bộ file, `MainActivity` không còn các field:

```text
ScientificEngine scientificEngine

TextView tvSciExpression
TextView tvSciResult
TextView tvSciAngleIndicator

String scientificInput
String scientificExpression
String scientificResult

boolean scientificJustEvaluated

ScientificPendingOperation scientificPendingOperation
double scientificPendingFirstValue
String scientificPendingFirstText
```

Không còn enum/interface:

```text
ScientificPendingOperation

ScientificUnaryOperation
```

Không còn các method Scientific:

```text
findScientificView()

setupScientificDigitButtons()
bindScientificDigitButton()
appendScientificDigit()

appendScientificDecimal()
getScientificCurrentNumberToken()

setupScientificOperatorButtons()
bindScientificOperatorButton()
appendScientificOperator()
removeTrailingScientificOperator()

setupScientificFunctionButtons()

performScientificUnary()
evaluateScientificCurrentInput()

beginScientificPendingOperation()
clearScientificPendingOperation()

calculateScientificExpression()

applyScientificResult()
saveScientificHistory()

appendScientificConstant()

appendScientificOpenParenthesis()
appendScientificCloseParenthesis()
countCharacter()

applyScientificPercent()
toggleScientificSign()
scientificBackspace()

prepareScientificForValueInput()
resetScientificState()
renderScientificState()
```

Các phần trên hiện nằm trong:

```text
ScientificCalculatorController.java
```

`MainActivity` chỉ còn:

```java
private ScientificCalculatorController
        scientificCalculatorController;
```

và:

```java
private void setupScientificController() {

    scientificCalculatorController =
            new ScientificCalculatorController(
                    scientificCalculatorLayout,
                    new ScientificEngine(),
                    historyManager
            );

    scientificCalculatorController.setup();
}
```

---

# 3. Build checkpoint

Chạy:

```text
Build
→ Make Project
```

Checklist:

```text
[ ] MainActivity.java compile

[ ] ScientificCalculatorController import resolve

[ ] ScientificEngine import resolve

[ ] Không còn duplicate Scientific listener

[ ] Không còn unresolved method:
    renderScientificState
    resetScientificState
    setupScientificDigitButtons

[ ] Basic Calculator compile

[ ] Mode Popup compile

[ ] History Bottom Sheet compile
```

## Lỗi thường gặp 1

```text
Cannot resolve symbol
ScientificCalculatorController
```

Kiểm tra package của controller:

```java
package com.example.calculator.ui.controller;
```

và import trong `MainActivity`:

```java
import com.example.calculator.ui.controller
        .ScientificCalculatorController;
```

Viết import trên một dòng trong source:

```java
import com.example.calculator.ui.controller.ScientificCalculatorController;
```

## Lỗi thường gặp 2

```text
Cannot resolve symbol rowModeConvert
```

Kiểm tra `popup_calculator_mode.xml` có:

```xml
android:id="@+id/rowModeConvert"
```

## Lỗi thường gặp 3

```text
NullPointerException
Scientific root view cannot be null
```

Kiểm tra `activity_main.xml` có:

```xml
<include
    android:id="@+id/scientificCalculatorLayout"
    layout="@layout/layout_scientific_calculator"
    ... />
```

## Lỗi thường gặp 4

```text
Cannot resolve symbol btn0
```

Kiểm tra ID trong `layout_basic_calculator.xml`.

Bản lịch sử project sử dụng:

```text
btn0
btn1
...
btn9
```

Nếu XML hiện tại dùng tên khác, sửa mapping trong:

```java
setupDigitButtons()
```

để khớp ID thật.

---

# 4. Runtime regression test

Sau khi Build pass, chạy app.

## Test A — Basic

```text
2 + 3 =
Expected: 5
```

```text
25 × 4 =
Expected: 100
```

```text
5 ÷ 0 =
Expected: Error
```

```text
0.1 + 0.2 =
Expected: 0.3
```

## Test B — Scientific

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

```text
100
log

Expected:
2
```

## Test C — DEG/RAD

```text
DEG
→ tap Deg
→ RAD
```

Nhập:

```text
1.570796326795
sin
```

Expected gần:

```text
1
```

## Test D — History

Sau khi thực hiện Basic và Scientific:

```text
History
```

Expected có cả hai nhóm calculation:

```text
2 + 3
5

√(16)
4
```

## Test E — Mode

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
Top bar vẫn hiển thị
Popup có đúng một checkmark
```

---

# 5. Kết quả sau R1B

```text
MainActivity
│
├── Lifecycle
├── System bars
├── Basic Calculator
├── History launcher
├── Mode popup
├── Mode switching
└── Scientific controller initialization


ScientificCalculatorController
│
├── Scientific state
├── Scientific button listeners
├── Scientific input
├── Scientific render
├── DEG/RAD
├── Scientific History
└── ScientificEngine calls
```

Trạng thái:

```text
[✓] R1A
    Tạo ScientificCalculatorController

[~] R1B
    Nối controller vào MainActivity
    Xóa Scientific code khỏi MainActivity
    Chờ Build + Runtime test

NEXT

R2A
→ tạo BasicCalculatorController
→ chưa sửa MainActivity
→ Build
```
