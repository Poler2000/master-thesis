module Logger

export log, LOG_TO_CONSOLE, LOGFILE

LOG_TO_CONSOLE = true
LOGFILE = Nothing

function log_msg(msg::String)
    if LOG_TO_CONSOLE
        println(msg)
    end

    if LOGFILE != Nothing
        open(LOGFILE, "a") do file
            write(file, msg)
        end
    end
end

end